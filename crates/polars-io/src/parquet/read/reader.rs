use std::io::{Read, Seek};
use std::sync::Arc;

use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::*;
use polars_parquet::read;

use super::read_impl::read_parquet;
use super::utils::{ensure_matching_dtypes_if_found, projected_arrow_schema_to_projection_indices};
use crate::RowIndex;
use crate::mmap::MmapBytesReader;
use crate::parquet::metadata::FileMetadataRef;
use crate::prelude::*;

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
    metadata: Option<FileMetadataRef>,
    hive_partition_columns: Option<Vec<Series>>,
    include_file_path: Option<(PlSmallStr, Arc<str>)>,
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

    /// Checks that the file contains all the columns in `projected_arrow_schema` with the same
    /// dtype, and sets the projection indices.
    pub fn with_arrow_schema_projection(
        mut self,
        first_schema: &Arc<ArrowSchema>,
        projected_arrow_schema: Option<&ArrowSchema>,
        allow_missing_columns: bool,
    ) -> PolarsResult<Self> {
        let slf_schema = self.schema()?;
        let slf_schema_width = slf_schema.len();

        if allow_missing_columns {
            // Must check the dtypes
            ensure_matching_dtypes_if_found(
                projected_arrow_schema.unwrap_or(first_schema.as_ref()),
                self.schema()?.as_ref(),
            )?;
            self.schema = Some(Arc::new(
                first_schema
                    .iter()
                    .map(|(name, field)| {
                        (name.clone(), slf_schema.get(name).unwrap_or(field).clone())
                    })
                    .collect(),
            ));
        }

        let schema = self.schema()?;

        (|| {
            if let Some(projected_arrow_schema) = projected_arrow_schema {
                self.projection = projected_arrow_schema_to_projection_indices(
                    schema.as_ref(),
                    projected_arrow_schema,
                )?;
            } else {
                if slf_schema_width > first_schema.len() {
                    polars_bail!(
                       SchemaMismatch:
                       "parquet file contained extra columns and no selection was given"
                    )
                }

                self.projection =
                    projected_arrow_schema_to_projection_indices(schema.as_ref(), first_schema)?;
            };
            Ok(())
        })()
        .map_err(|e| {
            if !allow_missing_columns && matches!(e, PolarsError::ColumnNotFound(_)) {
                e.wrap_msg(|s| {
                    format!(
                        "error with column selection, \
                        consider enabling `allow_missing_columns`: {}",
                        s
                    )
                })
            } else {
                e
            }
        })?;

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

    pub fn set_metadata(&mut self, metadata: FileMetadataRef) {
        self.metadata = Some(metadata);
    }

    pub fn get_metadata(&mut self) -> PolarsResult<&FileMetadataRef> {
        if self.metadata.is_none() {
            self.metadata = Some(Arc::new(read::read_metadata(&mut self.reader)?));
        }
        Ok(self.metadata.as_ref().unwrap())
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
            schema: None,
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
        let n_rows = metadata.num_rows.min(self.slice.0 + self.slice.1);

        if let Some(cols) = &self.columns {
            self.projection = Some(columns_to_projection(cols, schema.as_ref())?);
        }

        let mut df = read_parquet(
            self.reader,
            self.slice,
            self.projection.as_deref(),
            &schema,
            Some(metadata),
            self.parallel,
            self.row_index,
            self.hive_partition_columns.as_deref(),
        )?;

        if self.rechunk {
            df.as_single_chunk_par();
        };

        if let Some((col, value)) = &self.include_file_path {
            unsafe {
                df.with_column_unchecked(Column::new_scalar(
                    col.clone(),
                    Scalar::new(
                        DataType::String,
                        AnyValue::StringOwned(value.as_ref().into()),
                    ),
                    if df.width() > 0 { df.height() } else { n_rows },
                ))
            };
        }

        Ok(df)
    }
}
