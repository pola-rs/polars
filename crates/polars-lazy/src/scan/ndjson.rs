use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::RowIndex;
use polars_plan::plans::{DslPlan, FileScan};
use polars_plan::prelude::{FileScanOptions, NDJsonReadOptions};

use crate::prelude::LazyFrame;
use crate::scan::file_list_reader::LazyFileListReader;

#[derive(Clone)]
pub struct LazyJsonLineReader {
    pub(crate) paths: Arc<[PathBuf]>,
    pub(crate) batch_size: Option<NonZeroUsize>,
    pub(crate) low_memory: bool,
    pub(crate) rechunk: bool,
    pub(crate) schema: Option<SchemaRef>,
    pub(crate) row_index: Option<RowIndex>,
    pub(crate) infer_schema_length: Option<NonZeroUsize>,
    pub(crate) n_rows: Option<usize>,
    pub(crate) ignore_errors: bool,
}

impl LazyJsonLineReader {
    pub fn new_paths(paths: Arc<[PathBuf]>) -> Self {
        Self::new(PathBuf::new()).with_paths(paths)
    }

    pub fn new(path: impl AsRef<Path>) -> Self {
        LazyJsonLineReader {
            paths: Arc::new([path.as_ref().to_path_buf()]),
            batch_size: None,
            low_memory: false,
            rechunk: false,
            schema: None,
            row_index: None,
            infer_schema_length: NonZeroUsize::new(100),
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
    pub fn with_infer_schema_length(mut self, num_rows: Option<NonZeroUsize>) -> Self {
        self.infer_schema_length = num_rows;
        self
    }
    /// Set the JSON file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.schema = schema;
        self
    }

    /// Reduce memory usage at the expense of performance
    #[must_use]
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: Option<NonZeroUsize>) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl LazyFileListReader for LazyJsonLineReader {
    fn finish(self) -> PolarsResult<LazyFrame> {
        if !self.glob() {
            return self.finish_no_glob();
        }

        let paths = self.expand_paths(false)?.0;

        let file_options = FileScanOptions {
            n_rows: self.n_rows,
            with_columns: None,
            cache: false,
            row_index: self.row_index,
            rechunk: self.rechunk,
            file_counter: 0,
            hive_options: Default::default(),
        };

        let options = NDJsonReadOptions {
            n_threads: None,
            infer_schema_length: self.infer_schema_length,
            chunk_size: NonZeroUsize::new(1 << 18).unwrap(),
            low_memory: self.low_memory,
            ignore_errors: self.ignore_errors,
            schema: self.schema,
        };

        let scan_type = FileScan::NDJson { options };

        Ok(LazyFrame::from(DslPlan::Scan {
            paths,
            file_info: None,
            hive_parts: None,
            predicate: None,
            file_options,
            scan_type,
        }))
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let file_options = FileScanOptions {
            n_rows: self.n_rows,
            with_columns: None,
            cache: false,
            row_index: self.row_index.clone(),
            rechunk: self.rechunk,
            file_counter: 0,
            hive_options: Default::default(),
        };

        let options = NDJsonReadOptions {
            n_threads: None,
            infer_schema_length: self.infer_schema_length,
            chunk_size: NonZeroUsize::new(1 << 18).unwrap(),
            low_memory: self.low_memory,
            ignore_errors: self.ignore_errors,
            schema: self.schema,
        };

        let scan_type = FileScan::NDJson { options };

        Ok(LazyFrame::from(DslPlan::Scan {
            paths: self.paths,
            file_info: None,
            hive_parts: None,
            predicate: None,
            file_options,
            scan_type,
        }))
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn with_paths(mut self, paths: Arc<[PathBuf]>) -> Self {
        self.paths = paths;
        self
    }

    fn with_n_rows(mut self, n_rows: impl Into<Option<usize>>) -> Self {
        self.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.row_index = row_index.into();
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
