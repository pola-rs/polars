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
        if !self.args.glob {
            return self.finish_no_glob();
        }
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
