use std::path::PathBuf;

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::RowIndex;
use polars_plan::prelude::UnionArgs;

use crate::prelude::*;

/// Reads [LazyFrame] from a filesystem or a cloud storage.
/// Supports glob patterns.
///
/// Use [LazyFileListReader::finish] to get the final [LazyFrame].
pub trait LazyFileListReader: Clone {
    /// Get the final [LazyFrame].
    fn finish(self) -> PolarsResult<LazyFrame> {
        if !self.glob() {
            return self.finish_no_glob();
        }

        let lfs = self
            .paths()
            .iter()
            .map(|path| {
                self.clone()
                    // Each individual reader should not apply a row limit.
                    .with_n_rows(None)
                    // Each individual reader should not apply a row index.
                    .with_row_index(None)
                    .with_paths(Arc::new(vec![path.clone()]))
                    .with_rechunk(false)
                    .finish_no_glob()
                    .map_err(|e| {
                        polars_err!(
                            ComputeError: "error while reading {}: {}", path.display(), e
                        )
                    })
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        polars_ensure!(
            !lfs.is_empty(),
            ComputeError: "no matching files found in {:?}", self.paths().iter().map(|x| x.to_str().unwrap()).collect::<Vec<_>>()
        );

        let mut lf = self.concat_impl(lfs)?;
        if let Some(n_rows) = self.n_rows() {
            lf = lf.slice(0, n_rows as IdxSize)
        };
        if let Some(rc) = self.row_index() {
            lf = lf.with_row_index(&rc.name, Some(rc.offset))
        };

        Ok(lf)
    }

    /// Recommended concatenation of [LazyFrame]s from many input files.
    ///
    /// This method should not take into consideration [LazyFileListReader::n_rows]
    /// nor [LazyFileListReader::row_index].
    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        let args = UnionArgs {
            rechunk: self.rechunk(),
            parallel: true,
            to_supertypes: false,
            from_partitioned_ds: true,
            ..Default::default()
        };
        concat_impl(&lfs, args)
    }

    /// Get the final [LazyFrame].
    /// This method assumes, that path is *not* a glob.
    ///
    /// It is recommended to always use [LazyFileListReader::finish] method.
    fn finish_no_glob(self) -> PolarsResult<LazyFrame>;

    fn glob(&self) -> bool {
        true
    }

    fn paths(&self) -> &[PathBuf];

    /// Set paths of the scanned files.
    #[must_use]
    fn with_paths(self, paths: Arc<Vec<PathBuf>>) -> Self;

    /// Configure the row limit.
    fn with_n_rows(self, n_rows: impl Into<Option<usize>>) -> Self;

    /// Configure the row index.
    fn with_row_index(self, row_index: impl Into<Option<RowIndex>>) -> Self;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    fn rechunk(&self) -> bool;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(self, toggle: bool) -> Self;

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize>;

    /// Add a row index column.
    fn row_index(&self) -> Option<&RowIndex>;

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        None
    }
}
