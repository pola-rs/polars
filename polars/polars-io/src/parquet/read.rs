use std::io::{Read, Seek};
use std::sync::Arc;

use arrow::io::parquet::read;
use arrow::io::parquet::write::FileMetaData;
#[cfg(feature = "async")]
use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::read_impl::FetchRowGroupsFromMmapReader;
use crate::mmap::MmapBytesReader;
#[cfg(feature = "async")]
use crate::parquet::async_impl::FetchRowGroupsFromObjectStore;
#[cfg(feature = "async")]
use crate::parquet::async_impl::ParquetObjectStore;
use crate::parquet::read_impl::read_parquet;
pub use crate::parquet::read_impl::BatchedParquetReader;
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
use crate::RowCount;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParallelStrategy {
    /// Don't parallelize
    None,
    /// Parallelize over the columns
    Columns,
    /// Parallelize over the row groups
    RowGroups,
    /// Automatically determine over which unit to parallelize
    /// This will choose the most occurring unit.
    #[default]
    Auto,
}

/// Read Apache parquet format into a DataFrame.
#[must_use]
pub struct ParquetReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
    columns: Option<Vec<String>>,
    projection: Option<Vec<usize>>,
    parallel: ParallelStrategy,
    row_count: Option<RowCount>,
    low_memory: bool,
    metadata: Option<FileMetaData>,
}

impl<R: MmapBytesReader> ParquetReader<R> {
    #[cfg(feature = "lazy")]
    // todo! hoist to lazy crate
    pub fn _finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        projection: Option<&[usize]>,
    ) -> PolarsResult<DataFrame> {
        // this path takes predicates and parallelism into account
        let metadata = read::read_metadata(&mut self.reader)?;
        let schema = read::schema::infer_schema(&metadata)?;

        let rechunk = self.rechunk;
        read_parquet(
            self.reader,
            self.n_rows.unwrap_or(usize::MAX),
            projection,
            &schema,
            Some(metadata),
            predicate,
            self.parallel,
            self.row_count,
        )
        .map(|mut df| {
            if rechunk {
                df.rechunk();
            };
            df
        })
    }

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

    /// Stop parsing when `n` rows are parsed. By settings this parameter the csv will be parsed
    /// sequentially.
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

    /// Add a `row_count` column.
    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }

    /// [`Schema`] of the file.
    pub fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata()?;
        let schema = read::infer_schema(metadata)?;
        Ok(schema.fields.iter().into())
    }
    /// Number of rows in the parquet file.
    pub fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata()?;
        Ok(metadata.num_rows)
    }
    fn get_metadata(&mut self) -> PolarsResult<&FileMetaData> {
        if self.metadata.is_none() {
            self.metadata = Some(read::read_metadata(&mut self.reader)?);
        }
        Ok(self.metadata.as_ref().unwrap())
    }
}

impl<R: MmapBytesReader + 'static> ParquetReader<R> {
    pub fn batched(mut self, chunk_size: usize) -> PolarsResult<BatchedParquetReader> {
        let metadata = read::read_metadata(&mut self.reader)?;

        let row_group_fetcher = Box::new(FetchRowGroupsFromMmapReader::new(Box::new(self.reader))?);
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection,
            self.row_count,
            chunk_size,
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
            row_count: None,
            low_memory: false,
            metadata: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let metadata = read::read_metadata(&mut self.reader)?;
        let schema = read::schema::infer_schema(&metadata)?;

        if let Some(cols) = &self.columns {
            self.projection = Some(columns_to_projection(cols, &schema)?);
        }

        read_parquet(
            self.reader,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection.as_deref(),
            &schema,
            Some(metadata),
            None,
            self.parallel,
            self.row_count,
        )
        .map(|mut df| {
            if self.rechunk {
                df.rechunk();
            }
            df
        })
    }
}

/// A Parquet reader on top of the async object_store API. Only the batch reader is implemented since
/// parquet files on cloud storage tend to be big and slow to access.
#[cfg(feature = "async")]
pub struct ParquetAsyncReader {
    reader: ParquetObjectStore,
    rechunk: bool,
    n_rows: Option<usize>,
    projection: Option<Vec<usize>>,
    row_count: Option<RowCount>,
    low_memory: bool,
}

#[cfg(feature = "async")]
impl ParquetAsyncReader {
    pub fn from_uri(
        uri: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<ParquetAsyncReader> {
        Ok(ParquetAsyncReader {
            reader: ParquetObjectStore::from_uri(uri, cloud_options)?,
            rechunk: false,
            n_rows: None,
            projection: None,
            row_count: None,
            low_memory: false,
        })
    }

    /// Fetch the file info in a synchronous way to for the query planning phase.
    #[tokio::main(flavor = "current_thread")]
    pub async fn file_info(
        uri: &str,
        options: Option<&CloudOptions>,
    ) -> PolarsResult<(Schema, usize)> {
        let mut reader = ParquetAsyncReader::from_uri(uri, options)?;
        let schema = reader.schema().await?;
        let num_rows = reader.num_rows().await?;
        Ok((schema, num_rows))
    }

    pub async fn schema(&mut self) -> PolarsResult<Schema> {
        self.reader.schema().await
    }
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        self.reader.num_rows().await
    }

    pub fn with_n_rows(mut self, n_rows: Option<usize>) -> Self {
        self.n_rows = n_rows;
        self
    }

    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }

    pub fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    pub fn set_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    #[tokio::main(flavor = "current_thread")]
    pub async fn batched(mut self, chunk_size: usize) -> PolarsResult<BatchedParquetReader> {
        let metadata = self.reader.get_metadata().await?.to_owned();
        let row_group_fetcher = Box::new(FetchRowGroupsFromObjectStore::new(
            self.reader,
            &metadata,
            &self.projection,
        )?);
        BatchedParquetReader::new(
            row_group_fetcher,
            metadata,
            self.n_rows.unwrap_or(usize::MAX),
            self.projection,
            self.row_count,
            chunk_size,
        )
    }
}
