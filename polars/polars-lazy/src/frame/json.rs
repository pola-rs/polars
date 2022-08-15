use crate::prelude::{AnonymousScan, AnonymousScanOptions};
use polars_core::prelude::*;
use polars_io::prelude::{ndjson, JsonLineReader, StructArray};
use polars_io::{RowCount, SerReader};

use super::{LazyFrame, ScanArgsAnonymous};

pub struct LazyJsonReader {
    path: String,
    batch_size: Option<usize>,
    low_memory: bool,
    rechunk: bool,
    schema: Option<Schema>,
    row_count: Option<RowCount>,
    infer_schema_length: Option<usize>,
    skip_rows: Option<usize>,
    n_rows: Option<usize>,
}

impl LazyJsonReader {
    pub fn new(path: String) -> Self {
        LazyJsonReader {
            path,
            batch_size: None,
            low_memory: false,
            rechunk: true,
            schema: None,
            row_count: None,
            infer_schema_length: Some(100),
            skip_rows: None,
            n_rows: None,
        }
    }
    /// Add a `row_count` column.
    #[must_use]
    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }
    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    #[must_use]
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }
    /// Set the number of rows to use when inferring the json schema.
    /// the default is 100 rows.
    /// Setting to `None` will do a full table scan, very slow.
    #[must_use]
    pub fn with_infer_schema_length(mut self, num_rows: Option<usize>) -> Self {
        self.infer_schema_length = num_rows;
        self
    }
    /// Set the JSON file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Skip the first `n` rows during parsing. The header will be parsed at row `n`.
    #[must_use]
    pub fn with_skip_rows(mut self, skip_rows: Option<usize>) -> Self {
        self.skip_rows = skip_rows;
        self
    }
    /// Reduce memory usage in expensive of performance
    #[must_use]
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    pub fn with_rechunk(mut self, toggle: bool) -> Self {
        self.rechunk = toggle;
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn finish(self) -> Result<LazyFrame> {
        let options = ScanArgsAnonymous {
            name: "JSON SCAN",
            infer_schema_length: self.infer_schema_length,
            n_rows: self.n_rows,
            row_count: self.row_count.clone(),
            skip_rows: self.skip_rows.clone(),
            schema: self.schema.clone(),
            ..ScanArgsAnonymous::default()
        };

        LazyFrame::anonymous_scan(std::sync::Arc::new(self), options)
    }
}

impl AnonymousScan for LazyJsonReader {
    fn scan(&self, scan_opts: AnonymousScanOptions) -> Result<DataFrame> {
        let schema = scan_opts.output_schema.unwrap_or(scan_opts.schema);
        JsonLineReader::from_path(&self.path)?
            .with_schema(&schema)
            .with_rechunk(self.rechunk)
            .with_chunk_size(self.batch_size)
            .low_memory(self.low_memory)
            .with_n_rows(scan_opts.n_rows)
            .with_chunk_size(self.batch_size)
            .finish()
    }

    fn schema(&self, infer_schema_length: Option<usize>) -> Result<Schema> {
        let f = std::fs::File::open(&self.path)?;
        let mut reader = std::io::BufReader::new(f);

        let data_type = ndjson::read::infer(&mut reader, infer_schema_length)
            .map_err(|err| PolarsError::ComputeError(format!("{:#?}", err).into()))?;
        let schema: Schema = StructArray::get_fields(&data_type).into();

        Ok(schema)
    }
    fn allows_projection_pushdown(&self) -> bool {
        true
    }
}
