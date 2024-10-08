use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::read::ParallelStrategy;
use polars_io::{HiveOptions, RowIndex};

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub parallel: ParallelStrategy,
    pub row_index: Option<RowIndex>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,
    pub use_statistics: bool,
    pub schema: Option<SchemaRef>,
    pub low_memory: bool,
    pub rechunk: bool,
    pub cache: bool,
    /// Expand path given via globbing rules.
    pub glob: bool,
    pub include_file_paths: Option<PlSmallStr>,
    pub allow_missing_columns: bool,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            parallel: Default::default(),
            row_index: None,
            cloud_options: None,
            hive_options: Default::default(),
            use_statistics: true,
            schema: None,
            rechunk: false,
            low_memory: false,
            cache: true,
            glob: true,
            include_file_paths: None,
            allow_missing_columns: false,
        }
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    args: ScanArgsParquet,
    sources: ScanSources,
}

impl LazyParquetReader {
    fn new(args: ScanArgsParquet) -> Self {
        Self {
            args,
            sources: ScanSources::default(),
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    /// Get the final [LazyFrame].
    fn finish(self) -> PolarsResult<LazyFrame> {
        let row_index = self.args.row_index;

        let mut lf: LazyFrame = DslBuilder::scan_parquet(
            self.sources,
            self.args.n_rows,
            self.args.cache,
            self.args.parallel,
            None,
            self.args.rechunk,
            self.args.low_memory,
            self.args.cloud_options,
            self.args.use_statistics,
            self.args.schema,
            self.args.hive_options,
            self.args.glob,
            self.args.include_file_paths,
            self.args.allow_missing_columns,
        )?
        .build()
        .into();

        // It's a bit hacky, but this row_index function updates the schema.
        if let Some(row_index) = row_index {
            lf = lf.with_row_index(row_index.name.clone(), Some(row_index.offset))
        }

        lf.opt_state |= OptFlags::FILE_CACHING;
        Ok(lf)
    }

    fn glob(&self) -> bool {
        self.args.glob
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        unreachable!();
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

    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.args.cloud_options.as_ref()
    }

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_index(&self) -> Option<&RowIndex> {
        self.args.row_index.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        Self::scan_parquet_sources(
            ScanSources::Paths([path.as_ref().to_path_buf()].into()),
            args,
        )
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_sources(sources: ScanSources, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(args).with_sources(sources).finish()
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_files(paths: Arc<[PathBuf]>, args: ScanArgsParquet) -> PolarsResult<Self> {
        Self::scan_parquet_sources(ScanSources::Paths(paths), args)
    }
}
