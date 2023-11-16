use std::borrow::Cow;
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::ops::{Deref, Range};
use std::sync::Arc;

use arrow::array::new_empty_array;
use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use polars_parquet::read;
use polars_parquet::read::{ArrayIter, FileMetaData, RowGroupMetaData};
use rayon::prelude::*;

use super::mmap::ColumnStore;
use crate::mmap::{MmapBytesReader, ReaderBytes};
#[cfg(feature = "cloud")]
use crate::parquet::async_impl::FetchRowGroupsFromObjectStore;
use crate::parquet::mmap::mmap_columns;
use crate::parquet::predicates::read_this_row_group;
use crate::parquet::{mmap, FileMetaDataRef, ParallelStrategy};
use crate::predicates::{apply_predicate, PhysicalIoExpr};
use crate::utils::{apply_projection, get_reader_bytes};
use crate::RowCount;

fn enlarge_data_type(mut data_type: ArrowDataType) -> ArrowDataType {
    match data_type {
        ArrowDataType::Utf8 => {
            data_type = ArrowDataType::LargeUtf8;
        },
        ArrowDataType::Binary => {
            data_type = ArrowDataType::LargeBinary;
        },
        ArrowDataType::List(mut inner_field) => {
            inner_field.data_type = enlarge_data_type(inner_field.data_type);
            data_type = ArrowDataType::LargeList(inner_field);
        },
        _ => {},
    }
    data_type
}

fn column_idx_to_series(
    column_i: usize,
    md: &RowGroupMetaData,
    remaining_rows: usize,
    file_schema: &ArrowSchema,
    store: &mmap::ColumnStore,
    chunk_size: usize,
) -> PolarsResult<Series> {
    let mut field = file_schema.fields[column_i].clone();
    field.data_type = enlarge_data_type(field.data_type);

    let columns = mmap_columns(store, md.columns(), &field.name);
    let iter = mmap::to_deserializer(columns, field.clone(), remaining_rows, Some(chunk_size))?;

    if remaining_rows < md.num_rows() {
        array_iter_to_series(iter, &field, Some(remaining_rows))
    } else {
        array_iter_to_series(iter, &field, None)
    }
}

pub(super) fn array_iter_to_series(
    iter: ArrayIter,
    field: &ArrowField,
    num_rows: Option<usize>,
) -> PolarsResult<Series> {
    let mut total_count = 0;
    let chunks = match num_rows {
        None => iter.collect::<PolarsResult<Vec<_>>>()?,
        Some(n) => {
            let mut out = Vec::with_capacity(2);

            for arr in iter {
                let arr = arr?;
                let len = arr.len();
                out.push(arr);

                total_count += len;
                if total_count >= n {
                    break;
                }
            }
            out
        },
    };
    if chunks.is_empty() {
        let arr = new_empty_array(field.data_type.clone());
        Series::try_from((field.name.as_str(), arr))
    } else {
        Series::try_from((field.name.as_str(), chunks))
    }
}

/// Materializes hive partitions.
/// We have a special num_rows arg, as df can be empty when a projection contains
/// only hive partition columns.
/// Safety: num_rows equals the height of the df when the df height is non-zero.
pub(crate) fn materialize_hive_partitions(
    df: &mut DataFrame,
    hive_partition_columns: Option<&[Series]>,
    num_rows: usize,
) {
    if let Some(hive_columns) = hive_partition_columns {
        for s in hive_columns {
            unsafe { df.with_column_unchecked(s.new_from_index(0, num_rows)) };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rg_to_dfs(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
    row_count: Option<RowCount>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    if let ParallelStrategy::Columns | ParallelStrategy::None = parallel {
        rg_to_dfs_optionally_par_over_columns(
            store,
            previous_row_count,
            row_group_start,
            row_group_end,
            remaining_rows,
            file_metadata,
            schema,
            predicate,
            row_count,
            parallel,
            projection,
            use_statistics,
            hive_partition_columns,
        )
    } else {
        rg_to_dfs_par_over_rg(
            store,
            row_group_start,
            row_group_end,
            previous_row_count,
            remaining_rows,
            file_metadata,
            schema,
            predicate,
            row_count,
            projection,
            use_statistics,
            hive_partition_columns,
        )
    }
}

#[allow(clippy::too_many_arguments)]
// might parallelize over columns
fn rg_to_dfs_optionally_par_over_columns(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
    row_count: Option<RowCount>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    let mut dfs = Vec::with_capacity(row_group_end - row_group_start);

    for rg_idx in row_group_start..row_group_end {
        let md = &file_metadata.row_groups[rg_idx];
        let current_row_count = md.num_rows() as IdxSize;

        if use_statistics
            && !read_this_row_group(predicate, &file_metadata.row_groups[rg_idx], schema)?
        {
            *previous_row_count += current_row_count;
            continue;
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let projection_height = (*remaining_rows).min(md.num_rows());
        let chunk_size = md.num_rows();
        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                projection
                    .par_iter()
                    .map(|column_i| {
                        column_idx_to_series(
                            *column_i,
                            md,
                            projection_height,
                            schema,
                            store,
                            chunk_size,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            projection
                .iter()
                .map(|column_i| {
                    column_idx_to_series(
                        *column_i,
                        md,
                        projection_height,
                        schema,
                        store,
                        chunk_size,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        *remaining_rows -= projection_height;

        let mut df = DataFrame::new_no_checks(columns);
        if let Some(rc) = &row_count {
            df.with_row_count_mut(&rc.name, Some(*previous_row_count + rc.offset));
        }

        materialize_hive_partitions(&mut df, hive_partition_columns, projection_height);
        apply_predicate(&mut df, predicate, true)?;

        *previous_row_count += current_row_count;
        dfs.push(df);

        if *remaining_rows == 0 {
            break;
        }
    }
    Ok(dfs)
}

#[allow(clippy::too_many_arguments)]
// parallelizes over row groups
fn rg_to_dfs_par_over_rg(
    store: &mmap::ColumnStore,
    row_group_start: usize,
    row_group_end: usize,
    previous_row_count: &mut IdxSize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
    row_count: Option<RowCount>,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    // compute the limits per row group and the row count offsets
    let row_groups = file_metadata
        .row_groups
        .iter()
        .enumerate()
        .skip(row_group_start)
        .take(row_group_end - row_group_start)
        .map(|(rg_idx, rg_md)| {
            let row_count_start = *previous_row_count;
            let num_rows = rg_md.num_rows();
            *previous_row_count += num_rows as IdxSize;
            let projection_height = (*remaining_rows).min(num_rows);
            *remaining_rows -= projection_height;

            (rg_idx, rg_md, projection_height, row_count_start)
        })
        .collect::<Vec<_>>();

    let dfs = row_groups
        .into_par_iter()
        .map(|(rg_idx, md, projection_height, row_count_start)| {
            if projection_height == 0
                || use_statistics
                    && !read_this_row_group(predicate, &file_metadata.row_groups[rg_idx], schema)?
            {
                return Ok(None);
            }
            // test we don't read the parquet file if this env var is set
            #[cfg(debug_assertions)]
            {
                assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
            }

            let chunk_size = md.num_rows();
            let columns = projection
                .iter()
                .map(|column_i| {
                    column_idx_to_series(
                        *column_i,
                        md,
                        projection_height,
                        schema,
                        store,
                        chunk_size,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut df = DataFrame::new_no_checks(columns);

            if let Some(rc) = &row_count {
                df.with_row_count_mut(&rc.name, Some(row_count_start as IdxSize + rc.offset));
            }

            materialize_hive_partitions(&mut df, hive_partition_columns, projection_height);
            apply_predicate(&mut df, predicate, false)?;

            Ok(Some(df))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(dfs.into_iter().flatten().collect())
}

pub(super) fn materialize_empty_df(
    projection: Option<&[usize]>,
    reader_schema: &ArrowSchema,
    hive_partition_columns: Option<&[Series]>,
    row_count: Option<&RowCount>,
) -> DataFrame {
    let schema = if let Some(projection) = projection {
        Cow::Owned(apply_projection(reader_schema, projection))
    } else {
        Cow::Borrowed(reader_schema)
    };
    let mut df = DataFrame::from(schema.as_ref());

    if let Some(row_count) = row_count {
        df.insert_at_idx(0, Series::new_empty(&row_count.name, &IDX_DTYPE))
            .unwrap();
    }

    materialize_hive_partitions(&mut df, hive_partition_columns, 0);

    df
}

#[allow(clippy::too_many_arguments)]
pub fn read_parquet<R: MmapBytesReader>(
    mut reader: R,
    mut limit: usize,
    projection: Option<&[usize]>,
    reader_schema: &ArrowSchemaRef,
    metadata: Option<FileMetaDataRef>,
    predicate: Option<&dyn PhysicalIoExpr>,
    mut parallel: ParallelStrategy,
    row_count: Option<RowCount>,
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<DataFrame> {
    // Fast path.
    if limit == 0 {
        return Ok(materialize_empty_df(
            projection,
            reader_schema,
            hive_partition_columns,
            row_count.as_ref(),
        ));
    }

    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader).map(Arc::new))?;
    let n_row_groups = file_metadata.row_groups.len();

    // if there are multiple row groups and categorical data
    // we need a string cache
    // we keep it alive until the end of the function
    let _sc = if n_row_groups > 1 {
        #[cfg(feature = "dtype-categorical")]
        {
            Some(polars_core::StringCacheHolder::hold())
        }
        #[cfg(not(feature = "dtype-categorical"))]
        {
            Some(0u8)
        }
    } else {
        None
    };

    let materialized_projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..reader_schema.len()).collect::<Vec<_>>()));

    if let ParallelStrategy::Auto = parallel {
        if n_row_groups > materialized_projection.len() || n_row_groups > POOL.current_num_threads()
        {
            parallel = ParallelStrategy::RowGroups;
        } else {
            parallel = ParallelStrategy::Columns;
        }
    }

    if let (ParallelStrategy::Columns, true) = (parallel, materialized_projection.len() == 1) {
        parallel = ParallelStrategy::None;
    }

    let reader = ReaderBytes::from(&reader);
    let bytes = reader.deref();
    let store = mmap::ColumnStore::Local(bytes);

    let dfs = rg_to_dfs(
        &store,
        &mut 0,
        0,
        n_row_groups,
        &mut limit,
        &file_metadata,
        reader_schema,
        predicate,
        row_count.clone(),
        parallel,
        &materialized_projection,
        use_statistics,
        hive_partition_columns,
    )?;

    if dfs.is_empty() {
        Ok(materialize_empty_df(
            projection,
            reader_schema,
            hive_partition_columns,
            row_count.as_ref(),
        ))
    } else {
        accumulate_dataframes_vertical(dfs)
    }
}

pub struct FetchRowGroupsFromMmapReader(ReaderBytes<'static>);

impl FetchRowGroupsFromMmapReader {
    pub fn new(mut reader: Box<dyn MmapBytesReader>) -> PolarsResult<Self> {
        // safety we will keep ownership on the struct and reference the bytes on the heap.
        // this should not work with passed bytes so we check if it is a file
        assert!(reader.to_file().is_some());
        let reader_ptr = unsafe {
            std::mem::transmute::<&mut dyn MmapBytesReader, &'static mut dyn MmapBytesReader>(
                reader.as_mut(),
            )
        };
        let reader_bytes = get_reader_bytes(reader_ptr)?;
        Ok(FetchRowGroupsFromMmapReader(reader_bytes))
    }
    fn fetch_row_groups(&mut self, _row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        Ok(mmap::ColumnStore::Local(self.0.deref()))
    }
}

// We couldn't use a trait as async trait gave very hard HRT lifetime errors.
// Maybe a puzzle for another day.
pub enum RowGroupFetcher {
    #[cfg(feature = "cloud")]
    ObjectStore(FetchRowGroupsFromObjectStore),
    Local(FetchRowGroupsFromMmapReader),
}

#[cfg(feature = "cloud")]
impl From<FetchRowGroupsFromObjectStore> for RowGroupFetcher {
    fn from(value: FetchRowGroupsFromObjectStore) -> Self {
        RowGroupFetcher::ObjectStore(value)
    }
}

impl From<FetchRowGroupsFromMmapReader> for RowGroupFetcher {
    fn from(value: FetchRowGroupsFromMmapReader) -> Self {
        RowGroupFetcher::Local(value)
    }
}

impl RowGroupFetcher {
    async fn fetch_row_groups(&mut self, _row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        match self {
            RowGroupFetcher::Local(f) => f.fetch_row_groups(_row_groups),
            #[cfg(feature = "cloud")]
            RowGroupFetcher::ObjectStore(f) => f.fetch_row_groups(_row_groups).await,
        }
    }
}

pub(super) fn compute_row_group_range(
    row_group_start: usize,
    row_group_end: usize,
    limit: usize,
    row_groups: &[RowGroupMetaData],
) -> usize {
    let mut row_group_end_truncated = row_group_start;
    let mut acc_row_count = 0;

    #[allow(clippy::needless_range_loop)]
    for rg_i in row_group_start..(std::cmp::min(row_group_end, row_groups.len())) {
        if acc_row_count >= limit {
            break;
        }
        row_group_end_truncated = rg_i + 1;
        acc_row_count += row_groups[rg_i].num_rows();
    }
    row_group_end_truncated
}

pub struct BatchedParquetReader {
    // use to keep ownership
    #[allow(dead_code)]
    row_group_fetcher: RowGroupFetcher,
    limit: usize,
    projection: Vec<usize>,
    schema: ArrowSchemaRef,
    metadata: FileMetaDataRef,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_count: Option<RowCount>,
    rows_read: IdxSize,
    row_group_offset: usize,
    n_row_groups: usize,
    chunks_fifo: VecDeque<DataFrame>,
    parallel: ParallelStrategy,
    chunk_size: usize,
    use_statistics: bool,
    hive_partition_columns: Option<Vec<Series>>,
    /// Has returned at least one materialized frame.
    has_returned: bool,
}

impl BatchedParquetReader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_group_fetcher: RowGroupFetcher,
        metadata: FileMetaDataRef,
        schema: ArrowSchemaRef,
        limit: usize,
        projection: Option<Vec<usize>>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        row_count: Option<RowCount>,
        chunk_size: usize,
        use_statistics: bool,
        hive_partition_columns: Option<Vec<Series>>,
    ) -> PolarsResult<Self> {
        let n_row_groups = metadata.row_groups.len();
        let projection = projection.unwrap_or_else(|| (0usize..schema.len()).collect::<Vec<_>>());

        let parallel =
            if n_row_groups > projection.len() || n_row_groups > POOL.current_num_threads() {
                ParallelStrategy::RowGroups
            } else {
                ParallelStrategy::Columns
            };

        Ok(BatchedParquetReader {
            row_group_fetcher,
            limit,
            projection,
            schema,
            metadata,
            row_count,
            rows_read: 0,
            predicate,
            row_group_offset: 0,
            n_row_groups,
            chunks_fifo: VecDeque::with_capacity(POOL.current_num_threads()),
            parallel,
            chunk_size,
            use_statistics,
            hive_partition_columns,
            has_returned: false,
        })
    }

    pub fn limit_reached(&self) -> bool {
        self.limit == 0
    }

    pub fn schema(&self) -> &ArrowSchemaRef {
        &self.schema
    }

    pub fn is_finished(&self) -> bool {
        self.row_group_offset >= self.n_row_groups
    }

    pub fn finishes_this_batch(&self, n: usize) -> bool {
        self.row_group_offset + n > self.n_row_groups
    }

    pub async fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if self.limit == 0 && self.has_returned {
            return Ok(None);
        }

        let mut skipped_all_rgs = false;
        // fill up fifo stack
        if self.row_group_offset < self.n_row_groups && self.chunks_fifo.len() < n {
            // Ensure we apply the limit on the metadata, before we download the row-groups.
            let row_group_start = self.row_group_offset;
            let row_group_end = compute_row_group_range(
                row_group_start,
                row_group_start + n,
                self.limit,
                &self.metadata.row_groups,
            );

            let store = self
                .row_group_fetcher
                .fetch_row_groups(row_group_start..row_group_end)
                .await?;

            let dfs = rg_to_dfs(
                &store,
                &mut self.rows_read,
                row_group_start,
                row_group_end,
                &mut self.limit,
                &self.metadata,
                &self.schema,
                self.predicate.as_deref(),
                self.row_count.clone(),
                self.parallel,
                &self.projection,
                self.use_statistics,
                self.hive_partition_columns.as_deref(),
            )?;

            self.row_group_offset += n;

            // case where there is no data in the file
            // the streaming engine needs at least a single chunk
            if self.rows_read == 0 && dfs.is_empty() {
                return Ok(Some(vec![materialize_empty_df(
                    Some(&self.projection),
                    self.schema.as_ref(),
                    self.hive_partition_columns.as_deref(),
                    self.row_count.as_ref(),
                )]));
            }

            // TODO! this is slower than it needs to be
            // we also need to parallelize over row groups here.

            skipped_all_rgs |= dfs.is_empty();
            for mut df in dfs {
                // make sure that the chunks are not too large
                let n = df.shape().0 / self.chunk_size;
                if n > 1 {
                    for df in split_df(&mut df, n)? {
                        self.chunks_fifo.push_back(df)
                    }
                } else {
                    self.chunks_fifo.push_back(df)
                }
            }
        };

        if self.chunks_fifo.is_empty() {
            if skipped_all_rgs {
                Ok(Some(vec![materialize_empty_df(
                    Some(self.projection.as_slice()),
                    self.schema(),
                    self.hive_partition_columns.as_deref(),
                    self.row_count.as_ref(),
                )]))
            } else {
                Ok(None)
            }
        } else {
            let mut chunks = Vec::with_capacity(n);
            let mut i = 0;
            while let Some(df) = self.chunks_fifo.pop_front() {
                chunks.push(df);
                i += 1;
                if i == n {
                    break;
                }
            }

            self.has_returned |= true;
            Ok(Some(chunks))
        }
    }

    /// Turn the batched reader into an iterator.
    #[cfg(feature = "async")]
    pub fn iter(self, batches_per_iter: usize) -> BatchedParquetIter {
        BatchedParquetIter {
            batches_per_iter,
            inner: self,
            current_batch: vec![].into_iter(),
        }
    }
}

#[cfg(feature = "async")]
pub struct BatchedParquetIter {
    batches_per_iter: usize,
    inner: BatchedParquetReader,
    current_batch: std::vec::IntoIter<DataFrame>,
}

#[cfg(feature = "async")]
impl BatchedParquetIter {
    // todo! implement stream
    pub(crate) async fn next_(&mut self) -> Option<PolarsResult<DataFrame>> {
        match self.current_batch.next() {
            Some(df) => Some(Ok(df)),
            None => match self.inner.next_batches(self.batches_per_iter).await {
                Err(e) => Some(Err(e)),
                Ok(opt_batch) => {
                    let batch = opt_batch?;
                    self.current_batch = batch.into_iter();
                    self.current_batch.next().map(Ok)
                },
            },
        }
    }
}
