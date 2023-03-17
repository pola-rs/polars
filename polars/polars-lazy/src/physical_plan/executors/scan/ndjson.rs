use polars_core::error::to_compute_err;

use super::*;
use crate::prelude::{AnonymousScan, AnonymousScanOptions, LazyJsonLineReader};

impl AnonymousScan for LazyJsonLineReader {
    fn scan(&self, scan_opts: AnonymousScanOptions) -> PolarsResult<DataFrame> {
        let schema = scan_opts.output_schema.unwrap_or(scan_opts.schema);
        JsonLineReader::from_path(&self.path)?
            .with_schema(&schema)
            .with_rechunk(self.options.rechunk)
            .with_chunk_size(self.options.batch_size)
            .low_memory(self.options.low_memory)
            .with_n_rows(scan_opts.n_rows)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> PolarsResult<Schema> {
        let f = std::fs::File::open(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let data_type =
            arrow_ndjson::read::infer(&mut reader, infer_schema_length).map_err(to_compute_err)?;
        let schema: Schema = StructArray::get_fields(&data_type).iter().into();

        Ok(schema)
    }
    fn allows_projection_pushdown(&self) -> bool {
        true
    }
}
