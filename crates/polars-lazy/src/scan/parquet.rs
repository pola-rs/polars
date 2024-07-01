use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::read::ParallelStrategy;
use polars_io::utils::is_cloud_url;
use polars_io::{HiveOptions, RowIndex};

use crate::prelude::*;
use crate::scan::file_list_reader::get_glob_start_idx;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub parallel: ParallelStrategy,
    pub row_index: Option<RowIndex>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,
    pub use_statistics: bool,
    pub low_memory: bool,
    pub rechunk: bool,
    pub cache: bool,
    /// Expand path given via globbing rules.
    pub glob: bool,
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
            rechunk: false,
            low_memory: false,
            cache: true,
            glob: true,
        }
    }
}

impl ScanArgsParquet {
    /// Setter for the `n_rows` field.
    #[inline(always)]
    pub fn n_rows(mut self, n_rows: Option<usize>) -> Self {
        self.n_rows = n_rows;
        self
    }
    /// Setter for the `cache` field.
    #[inline(always)]
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }
    /// Setter for the `parallel` field.
    #[inline(always)]
    pub fn parallel(mut self, parallel: ParallelStrategy) -> Self {
        self.parallel = parallel;
        self
    }
    /// Setter for the `rechunk` field.
    #[inline(always)]
    pub fn rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }
    /// Setter for the `row_index` field.
    #[inline(always)]
    pub fn row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }
    /// Setter for the `low_memory` field.
    #[inline(always)]
    pub fn low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }
    /// Setter for the `cloud_options` field.
    #[inline(always)]
    pub fn cloud_options(mut self, cloud_options: Option<CloudOptions>) -> Self {
        self.cloud_options = cloud_options;
        self
    }
    /// Setter for the `use_statistics` field.
    #[inline(always)]
    pub fn use_statistics(mut self, use_statistics: bool) -> Self {
        self.use_statistics = use_statistics;
        self
    }
    /// Setter for the `hive_options` field.
    #[inline(always)]
    pub fn hive_options(mut self, hive_options: HiveOptions) -> Self {
        self.hive_options = hive_options;
        self
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    args: ScanArgsParquet,
    paths: Arc<[PathBuf]>,
}

impl LazyParquetReader {
    fn new(args: ScanArgsParquet) -> Self {
        Self {
            args,
            paths: Arc::new([]),
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    /// Get the final [LazyFrame].
    fn finish(mut self) -> PolarsResult<LazyFrame> {
        let (paths, hive_start_idx) =
            self.expand_paths(self.args.hive_options.enabled.unwrap_or(false))?;
        self.args.hive_options.enabled =
            Some(self.args.hive_options.enabled.unwrap_or_else(|| {
                self.paths.len() == 1
                    && get_glob_start_idx(self.paths[0].to_str().unwrap().as_bytes()).is_none()
                    && !paths.is_empty()
                    && {
                        (!is_cloud_url(&paths[0]) && paths[0].is_dir())
                            || (paths[0] != self.paths[0])
                    }
            }));
        self.args.hive_options.hive_start_idx = hive_start_idx;

        let row_index = self.args.row_index;

        let mut lf: LazyFrame = DslBuilder::scan_parquet(
            paths,
            self.args.n_rows,
            self.args.cache,
            self.args.parallel,
            None,
            self.args.rechunk,
            self.args.low_memory,
            self.args.cloud_options,
            self.args.use_statistics,
            self.args.hive_options,
        )?
        .build()
        .into();

        // it is a bit hacky, but this row_index function updates the schema
        if let Some(row_index) = row_index {
            lf = lf.with_row_index(&row_index.name, Some(row_index.offset))
        }

        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    fn glob(&self) -> bool {
        self.args.glob
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        unreachable!();
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
        LazyParquetReader::new(args)
            .with_paths(Arc::new([path.as_ref().to_path_buf()]))
            .finish()
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_files(paths: Arc<[PathBuf]>, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(args).with_paths(paths).finish()
    }
}
