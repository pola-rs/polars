use std::fmt::{Display, Formatter};

use arrow::io::parquet::read::FileMetaData;
use async_trait::async_trait;
use futures::Stream;
use opendal::Operator;
use polars_core::schema::Schema;
use polars_error::{to_compute_err, PolarsResult};

use crate::file_format::ObjectInfo;
use crate::input::file_format::FileFormat;

#[derive(Debug)]
pub struct ParquetFormat {}

impl ParquetFormat {
    /// Read and parse the schema of the Parquet file at location `path`
    async fn fetch_metadata_async(
        &self,
        operator: &Operator,
        path: impl AsRef<str>,
    ) -> PolarsResult<(FileMetaData, (usize, Option<usize>))> {
        let mut reader = operator
            .reader(path.as_ref())
            .await
            .map_err(to_compute_err)?;

        let metadata = arrow::io::parquet::read::read_metadata_async(&mut reader).await?;
        Ok((metadata, reader.size_hint()))
    }

    /// Parses the schema of the Parquet file from FileMetadata
    fn infer_schema(&self, metadata: FileMetaData) -> PolarsResult<Schema> {
        let arrow_schema =
            arrow::io::parquet::read::infer_schema(&metadata).map_err(to_compute_err)?;
        let polars_schema = Schema::from_iter(arrow_schema.clone().fields.iter());

        Ok(polars_schema)
    }
}

impl Display for ParquetFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParquetFormat()")
    }
}

#[async_trait]
impl FileFormat for ParquetFormat {
    /// Construct a new Format with no local overrides
    fn create() -> Self {
        Self {}
    }

    /// Uses a size hint obtained from the reader to produce,
    ///  - Known row count (may or may not be known)
    ///  - Estimated row count (can be calculated from reader hints)
    fn calculate_rows_count(
        &self,
        _reader_size_hint: (usize, Option<usize>),
    ) -> (Option<usize>, usize) {
        let (estimated, known) = _reader_size_hint;
        (known, estimated)
    }

    async fn get_object_info(&self, operator: &Operator, path: String) -> PolarsResult<ObjectInfo> {
        let (metadata, _) = self.fetch_metadata_async(operator, path.clone()).await?;
        let num_rows = &metadata.num_rows;
        let size_hint = self.calculate_rows_count((*num_rows, Some(*num_rows)));
        let polars_schema = self.infer_schema(metadata.clone())?;

        Ok((path.to_string(), polars_schema, size_hint))
    }
}
