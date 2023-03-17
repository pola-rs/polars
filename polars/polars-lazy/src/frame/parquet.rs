use std::path::{Path, PathBuf};

use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
struct LazyParquetReader {
    args: ParquetOptions,
    path: PathBuf,
}

impl LazyParquetReader {
    fn new(path: PathBuf, args: ParquetOptions) -> Self {
        Self { args, path }
    }
}

impl LazyFileListReader for LazyParquetReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let row_count = self.args.row_count.clone();

        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(self.path, self.args)?
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

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_count(&self) -> Option<&RowCount> {
        self.args.row_count.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    #[deprecated(note = "please use `concat_lf` instead")]
    pub fn scan_parquet_files<P: AsRef<Path>>(
        paths: Vec<P>,
        args: ParquetOptions,
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
    pub fn scan_parquet(path: impl AsRef<Path>, args: ParquetOptions) -> PolarsResult<Self> {
        LazyParquetReader::new(path.as_ref().to_owned(), args).finish()
    }
}
