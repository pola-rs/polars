use std::borrow::Cow;
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::ops::{Deref, Range};
use std::sync::Arc;

use arrow::array::new_empty_array;
use arrow::io::parquet::read;
use arrow::io::parquet::read::{ArrayIter, FileMetaData, RowGroupMetaData};
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use rayon::prelude::*;

use super::mmap::ColumnStore;
use crate::mmap::{MmapBytesReader, ReaderBytes};
#[cfg(feature = "async")]
use crate::parquet::async_impl::FetchRowGroupsFromObjectStore;
use crate::parquet::mmap::mmap_columns;
use crate::parquet::predicates::read_this_row_group;
use crate::parquet::{mmap, ParallelStrategy};
#[cfg(feature = "async")]
use crate::pl_async::get_runtime;
use crate::predicates::{apply_predicate, arrow_schema_to_empty_df, PhysicalIoExpr};
use crate::utils::{apply_projection, get_reader_bytes};
use crate::RowCount;

fn column_idx_to_series(
    column_i: usize,
    md: &RowGroupMetaData,
    remaining_rows: usize,
    schema: &ArrowSchema,
    store: &mmap::ColumnStore,
    chunk_size: usize,
) -> PolarsResult<Series> {
    let mut field = schema.fields[column_i].clone();

    match field.data_type {
        ArrowDataType::Utf8 => {
            field.data_type = ArrowDataType::LargeUtf8;
        },
        ArrowDataType::Binary => {
            field.data_type = ArrowDataType::LargeBinary;
        },
        ArrowDataType::List(fld) => field.data_type = ArrowDataType::LargeList(fld),
        _ => {},
    }

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
        None => iter.collect::<arrow::error::Result<Vec<_>>>()?,
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
/// We have a special num_rows arg, as df can be empty.
fn materialize_hive_partitions(
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
// might parallelize over columns
fn rg_to_dfs(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchema,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_count: Option<RowCount>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    let mut dfs = Vec::with_capacity(row_group_end - row_group_start);

    for rg in row_group_start..row_group_end {
        let md = &file_metadata.row_groups[rg];
        let current_row_count = md.num_rows() as IdxSize;

        if use_statistics && !read_this_row_group(predicate.as_ref(), file_metadata, schema, rg)? {
            *previous_row_count += current_row_count;
            continue;
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let chunk_size = md.num_rows();
        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                projection
                    .par_iter()
                    .map(|column_i| {
                        column_idx_to_series(
                            *column_i,
                            md,
                            *remaining_rows,
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
                    column_idx_to_series(*column_i, md, *remaining_rows, schema, store, chunk_size)
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        *remaining_rows = remaining_rows.saturating_sub(file_metadata.row_groups[rg].num_rows());

        let mut df = DataFrame::new_no_checks(columns);
        if let Some(rc) = &row_count {
            df.with_row_count_mut(&rc.name, Some(*previous_row_count + rc.offset));
        }
        materialize_hive_partitions(&mut df, hive_partition_columns, md.num_rows());

        apply_predicate(&mut df, predicate.as_deref(), true)?;

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
fn rg_to_dfs_par(
    store: &mmap::ColumnStore,
    row_group_start: usize,
    row_group_end: usize,
    previous_row_count: &mut IdxSize,
    remaining_rows: &mut usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchema,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
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
            let local_limit = *remaining_rows;
            *remaining_rows = remaining_rows.saturating_sub(num_rows);

            (rg_idx, rg_md, local_limit, row_count_start)
        })
        .collect::<Vec<_>>();

    let dfs = row_groups
        .into_par_iter()
        .map(|(rg_idx, md, local_limit, row_count_start)| {
            if local_limit == 0
                || use_statistics
                    && !read_this_row_group(predicate.as_ref(), file_metadata, schema, rg_idx)?
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
                    column_idx_to_series(*column_i, md, local_limit, schema, store, chunk_size)
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut df = DataFrame::new_no_checks(columns);

            if let Some(rc) = &row_count {
                df.with_row_count_mut(&rc.name, Some(row_count_start as IdxSize + rc.offset));
            }
            materialize_hive_partitions(&mut df, hive_partition_columns, md.num_rows());

            apply_predicate(&mut df, predicate.as_deref(), false)?;

            Ok(Some(df))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(dfs.into_iter().flatten().collect())
}

#[allow(clippy::too_many_arguments)]
pub fn read_parquet<R: MmapBytesReader>(
    mut reader: R,
    mut limit: usize,
    projection: Option<&[usize]>,
    schema: &ArrowSchema,
    metadata: Option<FileMetaData>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    mut parallel: ParallelStrategy,
    row_count: Option<RowCount>,
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<DataFrame> {
    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader))?;
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

    let projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..schema.fields.len()).collect::<Vec<_>>()));

    if let ParallelStrategy::Auto = parallel {
        if n_row_groups > projection.len() || n_row_groups > POOL.current_num_threads() {
            parallel = ParallelStrategy::RowGroups;
        } else {
            parallel = ParallelStrategy::Columns;
        }
    }

    if let (ParallelStrategy::Columns, true) = (parallel, projection.len() == 1) {
        parallel = ParallelStrategy::None;
    }

    let reader = ReaderBytes::from(&reader);
    let bytes = reader.deref();
    let store = mmap::ColumnStore::Local(bytes);
    let dfs = match parallel {
        ParallelStrategy::Columns | ParallelStrategy::None => rg_to_dfs(
            &store,
            &mut 0,
            0,
            n_row_groups,
            &mut limit,
            &file_metadata,
            schema,
            predicate,
            row_count,
            parallel,
            &projection,
            use_statistics,
            hive_partition_columns,
        )?,
        ParallelStrategy::RowGroups => rg_to_dfs_par(
            &store,
            0,
            file_metadata.row_groups.len(),
            &mut 0,
            &mut limit,
            &file_metadata,
            schema,
            predicate,
            row_count,
            &projection,
            use_statistics,
            hive_partition_columns,
        )?,
        // auto should already be replaced by Columns or RowGroups
        ParallelStrategy::Auto => unimplemented!(),
    };

    if dfs.is_empty() {
        let schema = if let Cow::Borrowed(_) = projection {
            Cow::Owned(apply_projection(schema, &projection))
        } else {
            Cow::Borrowed(schema)
        };
        Ok(arrow_schema_to_empty_df(&schema))
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
    async fn fetch_row_groups(&mut self, _row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        Ok(mmap::ColumnStore::Local(self.0.deref()))
    }
}

// We couldn't use a trait as async trait gave very hard HRT lifetime errors.
// Maybe a puzzle for another day.
pub enum RowGroupFetcher {
    #[cfg(feature = "async")]
    ObjectStore(FetchRowGroupsFromObjectStore),
    Local(FetchRowGroupsFromMmapReader),
}

#[cfg(feature = "async")]
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
            RowGroupFetcher::Local(f) => f.fetch_row_groups(_row_groups).await,
            #[cfg(feature = "async")]
            RowGroupFetcher::ObjectStore(f) => f.fetch_row_groups(_row_groups).await,
        }
    }
}

pub struct BatchedParquetReader {
    // use to keep ownership
    #[allow(dead_code)]
    row_group_fetcher: RowGroupFetcher,
    limit: usize,
    projection: Vec<usize>,
    schema: ArrowSchema,
    metadata: Arc<FileMetaData>,
    row_count: Option<RowCount>,
    rows_read: IdxSize,
    row_group_offset: usize,
    n_row_groups: usize,
    chunks_fifo: VecDeque<DataFrame>,
    parallel: ParallelStrategy,
    chunk_size: usize,
    use_statistics: bool,
    hive_partition_columns: Option<Vec<Series>>,
}

impl BatchedParquetReader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_group_fetcher: RowGroupFetcher,
        metadata: Arc<FileMetaData>,
        limit: usize,
        projection: Option<Vec<usize>>,
        row_count: Option<RowCount>,
        chunk_size: usize,
        use_statistics: bool,
        hive_partition_columns: Option<Vec<Series>>,
    ) -> PolarsResult<Self> {
        let schema = read::schema::infer_schema(&metadata)?;
        let n_row_groups = metadata.row_groups.len();
        let projection =
            projection.unwrap_or_else(|| (0usize..schema.fields.len()).collect::<Vec<_>>());

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
            row_group_offset: 0,
            n_row_groups,
            chunks_fifo: VecDeque::with_capacity(POOL.current_num_threads()),
            parallel,
            chunk_size,
            use_statistics,
            hive_partition_columns,
        })
    }

    pub async fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        // fill up fifo stack
        if self.row_group_offset <= self.n_row_groups && self.chunks_fifo.len() < n {
            let row_group_start = self.row_group_offset;
            let row_group_end = std::cmp::min(self.row_group_offset + n, self.n_row_groups);
            let store = self
                .row_group_fetcher
                .fetch_row_groups(row_group_start..row_group_end)
                .await?;
            let dfs = match self.parallel {
                ParallelStrategy::Columns => {
                    let dfs = rg_to_dfs(
                        &store,
                        &mut self.rows_read,
                        row_group_start,
                        row_group_end,
                        &mut self.limit,
                        &self.metadata,
                        &self.schema,
                        None,
                        self.row_count.clone(),
                        ParallelStrategy::Columns,
                        &self.projection,
                        self.use_statistics,
                        self.hive_partition_columns.as_deref(),
                    )?;
                    self.row_group_offset += n;
                    dfs
                },
                ParallelStrategy::RowGroups => {
                    let dfs = rg_to_dfs_par(
                        &store,
                        self.row_group_offset,
                        std::cmp::min(self.row_group_offset + n, self.n_row_groups),
                        &mut self.rows_read,
                        &mut self.limit,
                        &self.metadata,
                        &self.schema,
                        None,
                        self.row_count.clone(),
                        &self.projection,
                        self.use_statistics,
                        self.hive_partition_columns.as_deref(),
                    )?;
                    self.row_group_offset += n;
                    dfs
                },
                _ => unimplemented!(),
            };
            // case where there is no data in the file
            // the streaming engine needs at least a single chunk
            if self.rows_read == 0 && dfs.is_empty() {
                let columns = self
                    .schema
                    .fields
                    .iter()
                    .map(|field| {
                        let dtype: DataType = (&field.data_type).into();
                        Series::new_empty(&field.name, &dtype)
                    })
                    .collect::<Vec<_>>();
                return Ok(Some(vec![DataFrame::new_no_checks(columns)]));
            }

            // TODO! this is slower than it needs to be
            // we also need to parallelize over row groups here.

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
            Ok(None)
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

#[cfg(feature = "async")]
impl Iterator for BatchedParquetIter {
    type Item = PolarsResult<DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        get_runtime().block_on(self.next_())
    }
}
