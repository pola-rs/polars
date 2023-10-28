use std::path::{Path, PathBuf};

use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::{is_cloud_url, RowCount};

use crate::prelude::*;

pub type PathIterator = Box<dyn Iterator<Item = PolarsResult<PathBuf>>>;

// cloud_options is used only with async feature
#[allow(unused_variables)]
fn polars_glob(pattern: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<PathIterator> {
    if is_cloud_url(pattern) {
        #[cfg(feature = "async")]
        {
            let paths = polars_io::async_glob(pattern, cloud_options)?;
            Ok(Box::new(paths.into_iter().map(|a| Ok(PathBuf::from(&a)))))
        }
        #[cfg(not(feature = "async"))]
        panic!("Feature `async` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let paths = glob::glob(pattern)
            .map_err(|_| polars_err!(ComputeError: "invalid glob pattern given"))?;
        let paths = paths.map(|v| v.map_err(to_compute_err));
        Ok(Box::new(paths))
    }
}

/// Reads [LazyFrame] from a filesystem or a cloud storage.
/// Supports glob patterns.
///
/// Use [LazyFileListReader::finish] to get the final [LazyFrame].
pub trait LazyFileListReader: Clone {
    /// Get the final [LazyFrame].
    fn finish(self) -> PolarsResult<LazyFrame> {
        if let Some(paths) = self.iter_paths()? {
            let lfs = paths
                .map(|r| {
                    let path = r?;
                    self.clone()
                        .with_path(path.clone())
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
                ComputeError: "no matching files found in {}", self.path().display()
            );

            let mut lf = self.concat_impl(lfs)?;
            if let Some(n_rows) = self.n_rows() {
                lf = lf.slice(0, n_rows as IdxSize)
            };
            if let Some(rc) = self.row_count() {
                lf = lf.with_row_count(&rc.name, Some(rc.offset))
            };

            Ok(lf)
        } else {
            self.finish_no_glob()
        }
    }

    /// Recommended concatenation of [LazyFrame]s from many input files.
    ///
    /// This method should not take into consideration [LazyFileListReader::n_rows]
    /// nor [LazyFileListReader::row_count].
    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        concat_impl(&lfs, self.rechunk(), true, true, false)
    }

    /// Get the final [LazyFrame].
    /// This method assumes, that path is *not* a glob.
    ///
    /// It is recommended to always use [LazyFileListReader::finish] method.
    fn finish_no_glob(self) -> PolarsResult<LazyFrame>;

    /// Path of the scanned file.
    /// It can be potentially a glob pattern.
    fn path(&self) -> &Path;

    fn paths(&self) -> &[PathBuf];

    /// Set path of the scanned file.
    /// Support glob patterns.
    #[must_use]
    fn with_path(self, path: PathBuf) -> Self;

    /// Set paths of the scanned files.
    /// Doesn't glob patterns.
    #[must_use]
    fn with_paths(self, paths: Arc<[PathBuf]>) -> Self;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    fn rechunk(&self) -> bool;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(self, toggle: bool) -> Self;

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize>;

    /// Add a `row_count` column.
    fn row_count(&self) -> Option<&RowCount>;

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        None
    }

    /// Get list of files referenced by this reader.
    ///
    /// Returns [None] if path is not a glob pattern.
    fn iter_paths(&self) -> PolarsResult<Option<PathIterator>> {
        let paths = self.paths();
        if paths.is_empty() {
            let path_str = self.path().to_string_lossy();
            if path_str.contains('*') || path_str.contains('?') || path_str.contains('[') {
                polars_glob(&path_str, self.cloud_options()).map(Some)
            } else {
                Ok(None)
            }
        } else {
            polars_ensure!(self.path().to_string_lossy() == "", InvalidOperation: "expected only a single path argument");
            // Lint is incorrect as we need static lifetime.
            #[allow(clippy::unnecessary_to_owned)]
            Ok(Some(Box::new(paths.to_vec().into_iter().map(Ok))))
        }
    }
}
