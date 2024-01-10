use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_core::config::verbose;
use polars_core::prelude::*;
#[cfg(feature = "cloud")]
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_parquet::read::{self, RowGroupMetaData};

#[cfg(feature = "cloud")]
use super::async_impl::FetchRowGroupsFromObjectStore;
#[cfg(feature = "cloud")]
use super::async_impl::ParquetObjectStore;
use super::mmap::ColumnStore;
use super::predicates::read_this_row_group;
pub use super::read_impl::BatchedParquetReader;
use super::read_impl::{
    column_idx_to_series, compute_row_group_range, read_parquet, FetchRowGroupsFromMmapReader, RowGroupFetcher
};
#[cfg(feature = "cloud")]
use super::utils::materialize_empty_df;
#[cfg(feature = "cloud")]
use crate::cloud::CloudOptions;
use crate::hive::materialize_hive_partitions;
use crate::mmap::MmapBytesReader;
use crate::parquet::metadata::FileMetaDataRef;
use crate::parquet::read::metadata::PartitionedColumnChunkMD;
use crate::predicates::{apply_predicate, PhysicalIoExpr};
use crate::prelude::*;
use crate::RowIndex;

/// Read Apache parquet format into a DataFrame.
#[must_use]
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    slice: (usize, usize),
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
    parallel: ParallelStrategy,
    schema: Option<ArrowSchemaRef>,
    row_index: Option<RowIndex>,
    low_memory: bool,
    metadata: Option<FileMetaDataRef>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    hive_partition_columns: Option<Vec<Series>>,
    include_file_path: Option<(PlSmallStr, Arc<str>)>,
    use_statistics: bool,
}

impl<R: MmapBytesReader> ParquetReader<R> {
    /// Try to reduce memory pressure at the expense of performance. If setting this does not reduce memory
    /// enough, turn off parallelization.
    pub fn set_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    /// Read the parquet file in parallel (default). The single threaded reader consumes less memory.
    pub fn read_parallel(mut self, parallel: ParallelStrategy) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_slice(mut self, slice: Option<(usize, usize)>) -> Self {
        self.slice = slice.unwrap_or((0, usize::MAX));
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Add a row index column.
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Ensure the schema of the file matches the given schema. Calling this
    /// after setting the projection will ensure only the projected indices
    /// are checked.
    pub fn check_schema(mut self, schema: &ArrowSchema) -> PolarsResult<Self> {
        let self_schema = self.schema()?;
        let self_schema = self_schema.as_ref();

        if let Some(ref projection) = self.projection {
            let projection = projection.as_slice();

            ensure_matching_schema(
                &schema.try_project(projection)?,
                &self_schema.try_project(projection)?,
            )?;
        } else {
            ensure_matching_schema(schema, self_schema)?;
        }

        Ok(self)
    }

    /// [`Schema`] of the file.
    pub fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        self.schema = Some(match &self.schema {
            Some(schema) => schema.clone(),
            None => {
                let metadata = self.get_metadata()?;
                Arc::new(read::infer_schema(metadata)?)
            },
        });

        Ok(self.schema.clone().unwrap())
    }

    /// Use statistics in the parquet to determine if pages
    /// can be skipped from reading.
    pub fn use_statistics(mut self, toggle: bool) -> Self {
        self.use_statistics = toggle;
        self
    }

    /// Number of rows in the parquet file.
    pub fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata()?;
        Ok(metadata.num_rows)
    }

    pub fn with_hive_partition_columns(mut self, columns: Option<Vec<Series>>) -> Self {
        self.hive_partition_columns = columns;
        self
    }

    pub fn with_include_file_path(
        mut self,
        include_file_path: Option<(PlSmallStr, Arc<str>)>,
    ) -> Self {
        self.include_file_path = include_file_path;
        self
    }

    pub fn get_metadata(&mut self) -> PolarsResult<&FileMetaDataRef> {
        if self.metadata.is_none() {
            self.metadata = Some(Arc::new(read::read_metadata(&mut self.reader)?));
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }
}

impl<R: MmapBytesReader + 'static> ParquetReader<R> {
    pub fn batched(mut self, chunk_size: usize) -> PolarsResult<BatchedParquetReader> {
        let metadata = self.get_metadata()?.clone();
        let schema = self.schema()?;
        let row_groups = metadata.row_groups.clone();

        // XXX: Can a parquet file starts at an offset?
        self.reader.seek(SeekFrom::Start(0))?;
        let row_group_fetcher = FetchRowGroupsFromMmapReader::new(Box::new(self.reader))?.into();
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            row_groups,
            schema,
            self.slice,
            self.projection,
            self.predicate.clone(),
            self.row_index,
            chunk_size,
            self.use_statistics,
            self.hive_partition_columns,
            self.include_file_path,
            self.parallel,
        )
    }
}

impl<R: MmapBytesReader> SerReader<R> for ParquetReader<R> {
    /// Create a new [`ParquetReader`] from an existing `Reader`.
    fn new(reader: R) -> Self {
        ParquetReader {
            reader,
            rechunk: false,
            slice: (0, usize::MAX),
            columns: None,
            projection: None,
            parallel: Default::default(),
            row_index: None,
            low_memory: false,
            metadata: None,
            predicate: None,
            schema: None,
            use_statistics: true,
            hive_partition_columns: None,
            include_file_path: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let schema = self.schema()?;
        let metadata = self.get_metadata()?.clone();
        let n_rows = metadata.num_rows;

        if let Some(cols) = &self.columns {
            self.projection = Some(columns_to_projection(cols, schema.as_ref())?);
        }

        let mut df = read_parquet(
            self.reader,
            self.slice,
            self.projection.as_deref(),
            &schema,
            Some(metadata),
            None,
            self.predicate.as_deref(),
            self.parallel,
            self.row_index,
            self.use_statistics,
            self.hive_partition_columns.as_deref(),
        )?;

        if self.rechunk {
            df.as_single_chunk_par();
        };

        if let Some((col, value)) = &self.include_file_path {
            unsafe {
                df.with_column_unchecked(
                    StringChunked::full(
                        col.clone(),
                        value,
                        if df.width() > 0 { df.height() } else { n_rows },
                    )
                    .into_series(),
                )
            };
        }

        Ok(df)
    }
}

/// A Parquet reader on top of the async object_store API. Only the batch reader is implemented since
/// parquet files on cloud storage tend to be big and slow to access.
#[cfg(feature = "cloud")]
pub struct ParquetAsyncReader {
    reader: ParquetObjectStore,
    slice: (usize, usize),
    rechunk: bool,
    projection: Option<Vec<usize>>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_index: Option<RowIndex>,
    use_statistics: bool,
    hive_partition_columns: Option<Vec<Series>>,
    include_file_path: Option<(PlSmallStr, Arc<str>)>,
    schema: Option<ArrowSchemaRef>,
    parallel: ParallelStrategy,
    row_groups: Option<Vec<RowGroupMetaData>>,
}

#[cfg(feature = "cloud")]
impl ParquetAsyncReader {
    pub async fn from_uri(
        uri: &str,
        cloud_options: Option<&CloudOptions>,
        metadata: Option<FileMetaDataRef>,
    ) -> PolarsResult<ParquetAsyncReader> {
        Ok(ParquetAsyncReader {
            reader: ParquetObjectStore::from_uri(uri, cloud_options, metadata).await?,
            rechunk: false,
            slice: (0, usize::MAX),
            projection: None,
            row_index: None,
            predicate: None,
            use_statistics: true,
            hive_partition_columns: None,
            include_file_path: None,
            schema: None,
            parallel: Default::default(),
            row_groups: None,
        })
    }

    pub async fn check_schema(mut self, schema: &ArrowSchema) -> PolarsResult<Self> {
        let self_schema = self.schema().await?;
        let self_schema = self_schema.as_ref();

        if let Some(ref projection) = self.projection {
            let projection = projection.as_slice();

            ensure_matching_schema(
                &schema.try_project(projection)?,
                &self_schema.try_project(projection)?,
            )?;
        } else {
            ensure_matching_schema(schema, self_schema)?;
        }

        Ok(self)
    }

    pub async fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        self.schema = Some(match self.schema.as_ref() {
            Some(schema) => Arc::clone(schema),
            None => {
                let metadata = self.reader.get_metadata().await?;
                let arrow_schema = polars_parquet::arrow::read::infer_schema(metadata)?;
                Arc::new(arrow_schema)
            },
        });

        Ok(self.schema.clone().unwrap())
    }

    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        self.reader.num_rows().await
    }

    /// Only positive offsets are supported for simplicity - the caller should
    /// translate negative offsets into the positive equivalent.
    pub fn with_slice(mut self, slice: Option<(usize, usize)>) -> Self {
        self.slice = slice.unwrap_or((0, usize::MAX));
        self
    }

    pub async fn num_rows_with_predicate(&mut self) -> PolarsResult<usize> {
        let row_sizes = self
            .prune_row_groups(self.reader.clone())
            .await?
            .iter()
            .map(|(row_size, _md)| *row_size)
            .collect::<Vec<_>>();

        Ok(row_sizes.iter().sum())
    }

    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    pub fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }

    /// Use statistics in the parquet to determine if pages
    /// can be skipped from reading.
    pub fn use_statistics(mut self, toggle: bool) -> Self {
        self.use_statistics = toggle;
        self
    }

    pub fn with_hive_partition_columns(mut self, columns: Option<Vec<Series>>) -> Self {
        self.hive_partition_columns = columns;
        self
    }

    pub fn with_include_file_path(
        mut self,
        include_file_path: Option<(PlSmallStr, Arc<str>)>,
    ) -> Self {
        self.include_file_path = include_file_path;
        self
    }

    pub fn read_parallel(mut self, parallel: ParallelStrategy) -> Self {
        self.parallel = parallel;
        self
    }

    pub async fn batched(mut self, chunk_size: usize) -> PolarsResult<BatchedParquetReader> {
        let metadata = self.reader.get_metadata().await?.clone();
        let schema = match self.schema {
            Some(schema) => schema,
            None => self.schema().await?,
        };
        let mut row_groups = self
            .row_groups
            .clone()
            .unwrap_or(metadata.row_groups.clone());

        if self.slice.1 != 0 {
            self.slice = (0, usize::MAX);
        } else {
            row_groups = vec![];
        }
        // row group fetched deals with projection
        let row_group_fetcher = FetchRowGroupsFromObjectStore::new(
            self.reader,
            schema.clone(),
            self.projection.as_deref(),
            self.predicate.clone(),
            compute_row_group_range(0, metadata.row_groups.len(), self.slice, &row_groups),
            &row_groups,
        )?
        .into();
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            row_groups,
            schema,
            self.slice,
            self.projection,
            self.predicate.clone(),
            self.row_index,
            chunk_size,
            self.use_statistics,
            self.hive_partition_columns,
            self.include_file_path,
            self.parallel,
        )
    }

    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetaDataRef> {
        self.reader.get_metadata().await
    }

    pub async fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = self.get_metadata().await?.clone();
        let reader_schema = self.schema().await?;
        let row_index = self.row_index.clone();
        let hive_partition_columns = self.hive_partition_columns.clone();
        let projection = self.projection.clone();

        // batched reader deals with slice pushdown
        let reader = self.batched(usize::MAX).await?;
        let n_batches = metadata.row_groups.len();
        let mut iter = reader.iter(n_batches);

        let mut chunks = Vec::with_capacity(n_batches);
        while let Some(result) = iter.next_().await {
            chunks.push(result?)
        }
        if chunks.is_empty() {
            return Ok(materialize_empty_df(
                projection.as_deref(),
                reader_schema.as_ref(),
                hive_partition_columns.as_deref(),
                row_index.as_ref(),
            ));
        }
        let mut df = accumulate_dataframes_vertical_unchecked(chunks);

        if rechunk {
            df.as_single_chunk_par();
        }
        Ok(df)
    }

    #[cfg(feature = "cloud")]
    async fn prune_row_groups(
        &mut self,
        reader: ParquetObjectStore,
    ) -> PolarsResult<Vec<(usize, read::RowGroupMetaData)>> {


        let metadata = self.reader.get_metadata().await?.clone();
        let schema = &self.schema().await?;

        let predicate = self.predicate.clone();
        let projection = self.projection.as_deref();
        let hive_partition_columns = self.hive_partition_columns.as_deref();

        let predicate_columns = predicate.clone().unwrap().columns();
        let predicate_projection = materialize_projection(
            Some(&predicate_columns),
            &Schema::from_arrow_schema(&schema.clone()),
            hive_partition_columns,
            false,
        );

        let mut predicate_row_group_fetcher: RowGroupFetcher = FetchRowGroupsFromObjectStore::new(
            reader,
            schema.clone(),
            projection,
            predicate.clone(),
            compute_row_group_range(
                0,
                metadata.row_groups.len(),
                (0, usize::MAX),
                &metadata.row_groups,
            ),
            &metadata.row_groups,
        )?
        .into();

        let predicate_store: ColumnStore = predicate_row_group_fetcher
            .fetch_row_groups(0..metadata.row_groups.len())
            .await?;

        let mut remaining_rows = self.slice.1;

        let row_groups = metadata.row_groups.clone();
        let final_row_groups = row_groups
            .iter()
            .map(|md| {
                let mut part_md = PartitionedColumnChunkMD::new(md);
                let fields = predicate_columns.iter().map(|column| column.as_str()).collect::<PlHashSet<_>>();
                part_md.set_partitions(Some(&fields));

                if !read_this_row_group(predicate.clone().as_deref(), &part_md, schema).unwrap()
                    || remaining_rows == 0
                {
                    return (0, md);
                }

                let mut columns: Vec<Series> = vec![];


                for column_i in predicate_projection.as_ref().unwrap() {
                    let name = &schema.fields[*column_i].name;
                    let field_md = part_md.get_partitions(name).unwrap();
                    let column = column_idx_to_series(
                        *column_i,
                        field_md.as_slice(),
                        None,
                        &schema.clone(),
                        &predicate_store,
                    )
                    .unwrap();
                    columns.push(column)
                }

                let mut df = unsafe { DataFrame::new_no_checks(columns) };
                let reader_schema = schema.as_ref();

                materialize_hive_partitions(
                    &mut df,
                    reader_schema,
                    hive_partition_columns,
                    md.num_rows(),
                );
                apply_predicate(&mut df, predicate.as_deref(), false).unwrap();

                let row_count = df.height();

                remaining_rows = remaining_rows.saturating_sub(row_count);

                (row_count, md)
            })
            .filter(|(row_count, _md)| *row_count != 0)
            .map(|(row_count, md)| (row_count, md.clone()))
            .collect::<Vec<_>>();
        if verbose() {
            eprintln!(
                "reduced the number of row groups in pruning by {}",
                row_groups.len() - final_row_groups.len()
            )
        }
        let row_groups = Some(
            final_row_groups
                .clone()
                .into_iter()
                .map(|(_row_countm, row_group_metadata)| row_group_metadata)
                .collect::<Vec<_>>(),
        );
        self.row_groups = row_groups;
        Ok(final_row_groups)
    }
}
