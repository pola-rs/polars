use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::{HiveOptions, RowIndex};

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
    pub row_index: Option<RowIndex>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,
    pub include_file_paths: Option<PlSmallStr>,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_index: None,
            cloud_options: Default::default(),
            hive_options: Default::default(),
            include_file_paths: None,
        }
    }
}

#[derive(Clone)]
struct LazyIpcReader {
    args: ScanArgsIpc,
    sources: ScanSources,
}

impl LazyIpcReader {
    fn new(args: ScanArgsIpc) -> Self {
        Self {
            args,
            sources: ScanSources::default(),
        }
    }
}

impl LazyFileListReader for LazyIpcReader {
    fn finish(self) -> PolarsResult<LazyFrame> {
        let args = self.args;

        let options = IpcScanOptions {};

        let mut lf: LazyFrame = DslBuilder::scan_ipc(
            self.sources,
            options,
            args.n_rows,
            args.cache,
            args.row_index,
            args.rechunk,
            args.cloud_options,
            args.hive_options,
            args.include_file_paths,
        )?
        .build()
        .into();
        lf.opt_state |= OptFlags::FILE_CACHING;

        Ok(lf)
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        unreachable!()
    }

    fn sources(&self) -> &ScanSources {
        &self.sources
    }

    fn with_sources(mut self, sources: ScanSources) -> Self {
        self.sources = sources;
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

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.args.cloud_options.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        Self::scan_ipc_sources(
            ScanSources::Paths([path.as_ref().to_path_buf()].into()),
            args,
        )
    }

    pub fn scan_ipc_files(paths: Arc<[PathBuf]>, args: ScanArgsIpc) -> PolarsResult<Self> {
        Self::scan_ipc_sources(ScanSources::Paths(paths), args)
    }

    pub fn scan_ipc_sources(sources: ScanSources, args: ScanArgsIpc) -> PolarsResult<Self> {
        LazyIpcReader::new(args).with_sources(sources).finish()
    }
}
