use arrow::array::StructArray;
use async_trait::async_trait;
use opendal::Operator;
use polars_core::prelude::ArrowSchema;
use polars_error::{to_compute_err, PolarsResult};
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

use crate::input::files_async::{FileFormat, DEFAULT_SCHEMA_INFER_MAX_RECORD};

#[derive(Debug)]
pub struct NdJSONFormat {
    schema_infer_max_records: Option<usize>,
}

impl Default for NdJSONFormat {
    fn default() -> Self {
        Self {
            schema_infer_max_records: Some(DEFAULT_SCHEMA_INFER_MAX_RECORD),
        }
    }
}

impl NdJSONFormat {
    /// Construct a new Format with no local overrides
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a limit in terms of records to scan to infer the schema
    /// The default is `DEFAULT_SCHEMA_INFER_MAX_RECORD`
    pub fn set_schema_infer_max_records(mut self, schema_infer_max_records: Option<usize>) -> Self {
        self.schema_infer_max_records = schema_infer_max_records;
        self
    }
}

#[async_trait]
impl FileFormat for NdJSONFormat {
    /// Read and parse the schema of the ndjson/ jsonl file at location `path`
    async fn fetch_schema_async(
        &self,
        store_op: &Operator,
        path: String,
    ) -> PolarsResult<ArrowSchema> {
        let reader = store_op
            .reader(path.as_str())
            .await
            .map_err(to_compute_err)?;

        let mut stream_reader = StreamReader::new(reader);

        let mut line = String::new();
        let mut lines = Vec::new();

        loop {
            line.clear();
            let len = stream_reader
                .read_line(&mut line)
                .await
                .map_err(to_compute_err)?;
            if len == 0
                || lines.len()
                    >= self
                        .schema_infer_max_records
                        .unwrap_or(DEFAULT_SCHEMA_INFER_MAX_RECORD)
            {
                break;
            }

            lines.push(line.clone());
        }

        let dt = arrow::io::ndjson::read::infer_iter(lines.iter()).map_err(to_compute_err)?;
        let schema = ArrowSchema::from(StructArray::get_fields(&dt).to_vec());

        Ok(schema)
    }
}
