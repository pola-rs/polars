use std::path::{Path, PathBuf};

use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
use polars_io::parquet::ParallelStrategy;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: ParallelStrategy,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub low_memory: bool,
    pub cloud_options: Option<CloudOptions>,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            parallel: Default::default(),
            rechunk: true,
            row_count: None,
            low_memory: false,
            cloud_options: None,
        }
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    args: ScanArgsParquet,
    path: PathBuf,
}

impl LazyParquetReader {
    fn new(path: PathBuf, args: ScanArgsParquet) -> Self {
        Self { args, path }
    }
}

impl LazyFileListReader for LazyParquetReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let row_count = self.args.row_count;
        let path = self.path;
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(
            path,
            self.args.n_rows,
            self.args.cache,
            self.args.parallel,
            None,
            self.args.rechunk,
            self.args.low_memory,
            self.args.cloud_options,
        )?
        .build()
        .into();

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }

        lf.opt_state.file_caching = true;
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
        self.args.rechunk
    }

    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.args.rechunk = toggle;
        self
    }

    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.args.cloud_options.as_ref()
    }

    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        let args = &self.args;
        concat_impl(&lfs, args.rechunk, true, true).map(|mut lf| {
            if let Some(n_rows) = args.n_rows {
                lf = lf.slice(0, n_rows as IdxSize)
            };
            if let Some(rc) = args.row_count.clone() {
                lf = lf.with_row_count(&rc.name, Some(rc.offset))
            };
            lf
        })
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    #[deprecated(note = "please use `concat_lf` instead")]
    pub fn scan_parquet_files<P: AsRef<Path>>(
        paths: Vec<P>,
        args: ScanArgsParquet,
    ) -> PolarsResult<Self> {
        let reader = LazyParquetReader::new(
            paths.first().expect("got no files").as_ref().to_owned(),
            args,
        );
        let lfs = paths
            .iter()
            .map(|p| {
                reader
                    .clone()
                    .with_path(p.as_ref().to_owned())
                    .finish_no_glob()
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        reader.concat_impl(lfs)
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(path.as_ref().to_owned(), args).finish()
    }
}
