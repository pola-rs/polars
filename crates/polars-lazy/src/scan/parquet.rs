use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::ParallelStrategy;
use polars_io::{HiveOptions, RowIndex};

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: ParallelStrategy,
    pub rechunk: bool,
    pub row_index: Option<RowIndex>,
    pub low_memory: bool,
    pub cloud_options: Option<CloudOptions>,
    pub use_statistics: bool,
    pub hive_options: HiveOptions,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            parallel: Default::default(),
            rechunk: false,
            row_index: None,
            low_memory: false,
            cloud_options: None,
            use_statistics: true,
            hive_options: Default::default(),
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
    path: PathBuf,
    paths: Arc<[PathBuf]>,
}

impl LazyParquetReader {
    fn new(path: PathBuf, args: ScanArgsParquet) -> Self {
        Self {
            args,
            path,
            paths: Arc::new([]),
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    /// Get the final [LazyFrame].
    fn finish(mut self) -> PolarsResult<LazyFrame> {
        if let Some(paths) = self.iter_paths()? {
            let paths = paths
                .into_iter()
                .collect::<PolarsResult<Arc<[PathBuf]>>>()?;
            self.paths = paths;
        }
        self.finish_no_glob()
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let row_index = self.args.row_index;

        let paths = if self.paths.is_empty() {
            Arc::new([self.path]) as Arc<[PathBuf]>
        } else {
            self.paths
        };
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
        LazyParquetReader::new(path.as_ref().to_owned(), args).finish()
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_files(paths: Arc<[PathBuf]>, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(PathBuf::new(), args)
            .with_paths(paths)
            .finish()
    }
}
