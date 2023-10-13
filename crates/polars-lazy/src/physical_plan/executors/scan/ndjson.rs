use polars_core::error::to_compute_err;

use super::*;
use crate::prelude::{AnonymousScan, LazyJsonLineReader};

impl AnonymousScan for LazyJsonLineReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame> {
        let schema = scan_opts.output_schema.unwrap_or(scan_opts.schema);
        JsonLineReader::from_path(&self.path)?
            .with_schema(schema)
            .with_rechunk(self.rechunk)
            .with_chunk_size(self.batch_size)
            .low_memory(self.low_memory)
            .with_n_rows(scan_opts.n_rows)
            .with_chunk_size(self.batch_size)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> PolarsResult<SchemaRef> {
        // Short-circuit schema inference if the schema has been explicitly provided.
        if let Some(schema) = &self.schema {
            return Ok(schema.clone());
        }

        let f = polars_utils::open_file(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let data_type =
            polars_json::ndjson::infer(&mut reader, infer_schema_length).map_err(to_compute_err)?;
        let schema = Schema::from_iter(StructArray::get_fields(&data_type));

        Ok(Arc::new(schema))
    }
    fn allows_projection_pushdown(&self) -> bool {
        true
    }
}
