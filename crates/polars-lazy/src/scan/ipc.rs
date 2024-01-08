use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub memmap: bool,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: true,
            row_count: None,
            memmap: true,
        }
    }
}

#[derive(Clone)]
struct LazyIpcReader {
    args: ScanArgsIpc,
    path: PathBuf,
    paths: Arc<[PathBuf]>,
}

impl LazyIpcReader {
    fn new(path: PathBuf, args: ScanArgsIpc) -> Self {
        Self {
            args,
            path,
            paths: Arc::new([]),
        }
    }
}

impl LazyFileListReader for LazyIpcReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let args = self.args;
        let path = self.path;

        let options = IpcScanOptions {
            memmap: args.memmap,
        };
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(
            path,
            options,
            args.n_rows,
            args.cache,
            args.row_count.clone(),
            args.rechunk,
        )?
        .build()
        .into();
        lf.opt_state.file_caching = true;

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = args.row_count {
            lf = lf.with_row_index(&row_count.name, Some(row_count.offset))
        }

        Ok(lf)
    }

    fn path(&self) -> &Path {
        self.path.as_path()
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
        self.args.rechunk
    }

    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.args.rechunk = toggle;
        self
    }

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_count(&self) -> Option<&RowCount> {
        self.args.row_count.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(path.as_ref().to_owned(), args).finish()
    }

    pub fn scan_ipc_files(paths: Arc<[PathBuf]>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(PathBuf::new(), args)
            .with_paths(paths)
            .finish()
    }
}
