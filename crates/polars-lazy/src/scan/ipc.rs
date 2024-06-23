use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::RowIndex;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
    pub row_index: Option<RowIndex>,
    pub memory_map: bool,
    pub cloud_options: Option<CloudOptions>,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_index: None,
            memory_map: true,
            cloud_options: Default::default(),
        }
    }
}

#[derive(Clone)]
struct LazyIpcReader {
    args: ScanArgsIpc,
    paths: Arc<[PathBuf]>,
}

impl LazyIpcReader {
    fn new(args: ScanArgsIpc) -> Self {
        Self {
            args,
            paths: Arc::new([]),
        }
    }
}

impl LazyFileListReader for LazyIpcReader {
    fn finish(self) -> PolarsResult<LazyFrame> {
        let paths = self.expand_paths(false)?.0;
        let args = self.args;

        let options = IpcScanOptions {
            memory_map: args.memory_map,
        };

        let mut lf: LazyFrame = DslBuilder::scan_ipc(
            paths,
            options,
            args.n_rows,
            args.cache,
            args.row_index,
            args.rechunk,
            args.cloud_options,
        )?
        .build()
        .into();
        lf.opt_state.file_caching = true;

        Ok(lf)
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        unreachable!()
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn with_paths(mut self, paths: Arc<[PathBuf]>) -> Self {
        self.paths = paths;
        self
    }

    fn with_n_rows(mut self, n_rows: impl Into<Option<usize>>) -> Self {
        self.args.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.args.row_index = row_index.into();
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

    fn row_index(&self) -> Option<&RowIndex> {
        self.args.row_index.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(args)
            .with_paths(Arc::new([path.as_ref().to_path_buf()]))
            .finish()
    }

    pub fn scan_ipc_files(paths: Arc<[PathBuf]>, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(args).with_paths(paths).finish()
    }
}
