use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
struct LazyIpcReader {
    options: IpcOptions,
    path: PathBuf,
}

impl LazyIpcReader {
    fn new(path: PathBuf, options: IpcOptions) -> Self {
        Self { options, path }
    }
}

impl LazyFileListReader for LazyIpcReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let options = self.options;
        let path = self.path;

        let row_count = options.row_count.clone();

        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(path, options)?.build().into();
        lf.opt_state.file_caching = true;

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }

        Ok(lf)
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    fn rechunk(&self) -> bool {
        self.options.rechunk
    }

    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.options.rechunk = toggle;
        self
    }

    fn n_rows(&self) -> Option<usize> {
        self.options.n_rows
    }

    fn row_count(&self) -> Option<&RowCount> {
        self.options.row_count.as_ref()
    }
}

impl LazyFrame {
    /// Lazily read from an Arrow IPC (Feather v2) file or multiple files via glob patterns.
    /// 
    /// This allows the query optimizer to push down predicates and projections to the scan
    /// level, thereby potentially reducing memory overhead.
    pub fn scan_ipc(path: impl AsRef<Path>, options: IpcOptions) -> PolarsResult<Self> {
        LazyIpcReader::new(path.as_ref().to_owned(), options).finish()
    }
}
