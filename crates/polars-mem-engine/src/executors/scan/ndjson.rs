use std::num::NonZeroUsize;

use super::*;

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
            .with_ignore_errors(self.ignore_errors)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> PolarsResult<SchemaRef> {
        polars_ensure!(infer_schema_length != Some(0), InvalidOperation: "JSON requires positive 'infer_schema_length'");
        // Short-circuit schema inference if the schema has been explicitly provided,
        // or already inferred
        if let Some(schema) = &(*self.schema.read().unwrap()) {
            return Ok(schema.clone());
        }

        let f = polars_utils::open_file(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let schema = Arc::new(polars_io::ndjson::infer_schema(
            &mut reader,
            infer_schema_length.and_then(NonZeroUsize::new),
        )?);
        let mut guard = self.schema.write().unwrap();
        *guard = Some(schema.clone());

        Ok(schema)
    }
    fn allows_projection_pushdown(&self) -> bool {
        true
    }
}
