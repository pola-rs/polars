use std::io::{Read, Seek};
use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::*;
#[cfg(feature = "cloud")]
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_parquet::read;

#[cfg(feature = "cloud")]
use super::async_impl::FetchRowGroupsFromObjectStore;
#[cfg(feature = "cloud")]
use super::async_impl::ParquetObjectStore;
pub use super::read_impl::BatchedParquetReader;
use super::read_impl::{read_parquet, FetchRowGroupsFromMmapReader};
#[cfg(feature = "cloud")]
use super::utils::materialize_empty_df;
#[cfg(feature = "cloud")]
use crate::cloud::CloudOptions;
use crate::mmap::MmapBytesReader;
use crate::parquet::metadata::FileMetaDataRef;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
use crate::RowIndex;

/// Read Apache parquet format into a DataFrame.
#[must_use]
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
    parallel: ParallelStrategy,
    schema: Option<ArrowSchemaRef>,
    row_index: Option<RowIndex>,
    low_memory: bool,
    metadata: Option<FileMetaDataRef>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    hive_partition_columns: Option<Vec<Series>>,
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

    /// Stop reading at `num_rows` rows.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
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

    /// Set the [`Schema`] if already known. This must be exactly the same as
    /// the schema in the file itself.
    pub fn with_schema(mut self, schema: Option<ArrowSchemaRef>) -> Self {
        self.schema = schema;
        self
    }

    /// [`Schema`] of the file.
    pub fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        match &self.schema {
            Some(schema) => Ok(schema.clone()),
            None => {
                let metadata = self.get_metadata()?;
                Ok(Arc::new(read::infer_schema(metadata)?))
            },
        }
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

        let row_group_fetcher = FetchRowGroupsFromMmapReader::new(Box::new(self.reader))?.into();
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            schema,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection,
            self.predicate.clone(),
            self.row_index,
            chunk_size,
            self.use_statistics,
            self.hive_partition_columns,
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
            n_rows: None,
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
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let schema = self.schema()?;
        let metadata = self.get_metadata()?.clone();

        if let Some(cols) = &self.columns {
            self.projection = Some(columns_to_projection(cols, schema.as_ref())?);
        }

        read_parquet(
            self.reader,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection.as_deref(),
            &schema,
            Some(metadata),
            self.predicate.as_deref(),
            self.parallel,
            self.row_index,
            self.use_statistics,
            self.hive_partition_columns.as_deref(),
        )
        .map(|mut df| {
            if self.rechunk {
                df.as_single_chunk_par();
            }
            df
        })
    }
}

/// A Parquet reader on top of the async object_store API. Only the batch reader is implemented since
/// parquet files on cloud storage tend to be big and slow to access.
#[cfg(feature = "cloud")]
pub struct ParquetAsyncReader {
    reader: ParquetObjectStore,
    n_rows: Option<usize>,
    rechunk: bool,
    projection: Option<Vec<usize>>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_index: Option<RowIndex>,
    use_statistics: bool,
    hive_partition_columns: Option<Vec<Series>>,
    schema: Option<ArrowSchemaRef>,
    parallel: ParallelStrategy,
}

#[cfg(feature = "cloud")]
impl ParquetAsyncReader {
    pub async fn from_uri(
        uri: &str,
        cloud_options: Option<&CloudOptions>,
        schema: Option<ArrowSchemaRef>,
        metadata: Option<FileMetaDataRef>,
    ) -> PolarsResult<ParquetAsyncReader> {
        Ok(ParquetAsyncReader {
            reader: ParquetObjectStore::from_uri(uri, cloud_options, metadata).await?,
            rechunk: false,
            n_rows: None,
            projection: None,
            row_index: None,
            predicate: None,
            use_statistics: true,
            hive_partition_columns: None,
            schema,
            parallel: Default::default(),
        })
    }

    pub async fn schema(&mut self) -> PolarsResult<ArrowSchemaRef> {
        Ok(match self.schema.as_ref() {
            Some(schema) => Arc::clone(schema),
            None => {
                let metadata = self.reader.get_metadata().await?;
                let arrow_schema = polars_parquet::arrow::read::infer_schema(metadata)?;
                Arc::new(arrow_schema)
            },
        })
    }

    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        self.reader.num_rows().await
    }

    pub fn with_n_rows(mut self, n_rows: Option<usize>) -> Self {
        self.n_rows = n_rows;
        self
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
        // row group fetched deals with projection
        let row_group_fetcher = FetchRowGroupsFromObjectStore::new(
            self.reader,
            schema.clone(),
            self.projection.as_deref(),
            self.predicate.clone(),
            &metadata.row_groups,
            self.n_rows.unwrap_or(usize::MAX),
        )?
        .into();
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            schema,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection,
            self.predicate.clone(),
            self.row_index,
            chunk_size,
            self.use_statistics,
            self.hive_partition_columns,
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
}
