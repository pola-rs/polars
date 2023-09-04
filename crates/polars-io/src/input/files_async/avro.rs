use async_trait::async_trait;
use opendal::Operator;
use polars_core::prelude::ArrowSchema;
use polars_error::{to_compute_err, PolarsResult};

use crate::input::files_async::FileFormat;

#[derive(Debug, Default)]
pub struct AvroFormat {}

impl AvroFormat {
    /// Construct a new Format with no local overrides
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl FileFormat for AvroFormat {
    /// Read and parse the schema of the Avro file at location `path`
    async fn fetch_schema_async(
        &self,
        store_op: &Operator,
        path: String,
    ) -> PolarsResult<ArrowSchema> {
        let mut reader = store_op
            .reader(path.as_str())
            .await
            .map_err(to_compute_err)?;

        let metadata = arrow::io::avro::avro_schema::read_async::read_metadata(&mut reader)
            .await
            .map_err(to_compute_err)?;
        let schema =
            arrow::io::avro::read::infer_schema(&metadata.record).map_err(to_compute_err)?;

        Ok(schema)
    }
}
