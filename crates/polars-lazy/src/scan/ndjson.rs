use std::path::{Path, PathBuf};
use std::sync::RwLock;

use polars_core::prelude::*;
use polars_io::RowIndex;

use super::*;
use crate::prelude::{LazyFrame, ScanArgsAnonymous};

#[derive(Clone)]
pub struct LazyJsonLineReader {
    pub(crate) path: PathBuf,
    paths: Arc<[PathBuf]>,
    pub(crate) batch_size: Option<usize>,
    pub(crate) low_memory: bool,
    pub(crate) rechunk: bool,
    pub(crate) schema: Arc<RwLock<Option<SchemaRef>>>,
    pub(crate) row_index: Option<RowIndex>,
    pub(crate) infer_schema_length: Option<usize>,
    pub(crate) n_rows: Option<usize>,
    pub(crate) ignore_errors: bool,
}

impl LazyJsonLineReader {
    pub fn new_paths(paths: Arc<[PathBuf]>) -> Self {
        Self::new(PathBuf::new()).with_paths(paths)
    }

    pub fn new(path: impl AsRef<Path>) -> Self {
        LazyJsonLineReader {
            path: path.as_ref().to_path_buf(),
            paths: Arc::new([]),
            batch_size: None,
            low_memory: false,
            rechunk: false,
            schema: Arc::new(Default::default()),
            row_index: None,
            infer_schema_length: Some(100),
            ignore_errors: false,
            n_rows: None,
        }
    }
    /// Add a row index column.
    #[must_use]
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Set values as `Null` if parsing fails because of schema mismatches.
    #[must_use]
    pub fn with_ignore_errors(mut self, ignore_errors: bool) -> Self {
        self.ignore_errors = ignore_errors;
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
    /// Ignored when the schema is specified explicitly using [`Self::with_schema`].
    /// Setting to `None` will do a full table scan, very slow.
    #[must_use]
    pub fn with_infer_schema_length(mut self, num_rows: Option<usize>) -> Self {
        self.infer_schema_length = num_rows;
        self
    }
    /// Set the JSON file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.schema = Arc::new(RwLock::new(schema));
        self
    }

    /// Reduce memory usage in expensive of performance
    #[must_use]
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl LazyFileListReader for LazyJsonLineReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let options = ScanArgsAnonymous {
            name: "JSON SCAN",
            infer_schema_length: self.infer_schema_length,
            n_rows: self.n_rows,
            row_index: self.row_index.clone(),
            schema: self.schema.read().unwrap().clone(),
            ..ScanArgsAnonymous::default()
        };

        LazyFrame::anonymous_scan(std::sync::Arc::new(self), options)
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    fn with_paths(mut self, paths: Arc<[PathBuf]>) -> Self {
        self.paths = paths;
        self
    }

    fn rechunk(&self) -> bool {
        self.rechunk
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.rechunk = toggle;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize> {
        self.n_rows
    }

    /// Add a row index column.
    fn row_index(&self) -> Option<&RowIndex> {
        self.row_index.as_ref()
    }
}
