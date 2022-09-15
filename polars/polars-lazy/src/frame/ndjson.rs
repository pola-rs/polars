use polars_core::prelude::*;
use polars_io::RowCount;

use super::{LazyFrame, ScanArgsAnonymous};

pub struct LazyJsonLineReader {
    pub(crate) path: String,
    pub(crate) batch_size: Option<usize>,
    pub(crate) low_memory: bool,
    pub(crate) rechunk: bool,
    pub(crate) schema: Option<Schema>,
    pub(crate) row_count: Option<RowCount>,
    pub(crate) infer_schema_length: Option<usize>,
    pub(crate) n_rows: Option<usize>,
}

impl LazyJsonLineReader {
    pub fn new(path: String) -> Self {
        LazyJsonLineReader {
            path,
            batch_size: None,
            low_memory: false,
            rechunk: true,
            schema: None,
            row_count: None,
            infer_schema_length: Some(100),
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

    pub fn finish(self) -> PolarsResult<LazyFrame> {
        let options = ScanArgsAnonymous {
            name: "JSON SCAN",
            infer_schema_length: self.infer_schema_length,
            n_rows: self.n_rows,
            row_count: self.row_count.clone(),
            schema: self.schema.clone(),
            ..ScanArgsAnonymous::default()
        };

        LazyFrame::anonymous_scan(std::sync::Arc::new(self), options)
    }
}
