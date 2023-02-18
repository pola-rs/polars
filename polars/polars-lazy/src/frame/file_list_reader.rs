use std::path::{Path, PathBuf};

use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
use polars_io::is_cloud_url;

use crate::prelude::*;

pub type GlobIterator = Box<dyn Iterator<Item = PolarsResult<PathBuf>>>;

// cloud_options is used only with async feature
#[allow(unused_variables)]
fn polars_glob(pattern: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<GlobIterator> {
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
            .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;

        let paths = paths.map(|v| v.map_err(|e| PolarsError::ComputeError(format!("{e}").into())));

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
        if let Some(paths) = self.glob()? {
            let lfs = paths
                .map(|r| {
                    let path = r?;
                    self.clone()
                        .with_path(path.clone())
                        .with_rechunk(false)
                        .finish_no_glob()
                        .map_err(|e| {
                            PolarsError::ComputeError(
                                format!("while reading {} got {e:?}.", path.display()).into(),
                            )
                        })
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            if lfs.is_empty() {
                return PolarsResult::Err(PolarsError::ComputeError(
                    format!(
                        "no matching files found in {}",
                        self.path().to_string_lossy()
                    )
                    .into(),
                ));
            }

            self.concat_impl(lfs)
        } else {
            self.finish_no_glob()
        }
    }

    /// Recommended concatenation of [LazyFrame]s from many input files.
    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame>;

    /// Get the final [LazyFrame].
    /// This method assumes, that path is *not* a glob.
    ///
    /// It is recommended to always use [LazyFileListReader::finish] method.
    fn finish_no_glob(self) -> PolarsResult<LazyFrame>;

    /// Path of the scanned file.
    /// It can be potentially a glob pattern.
    fn path(&self) -> &Path;

    /// Set path of the scanned file.
    /// Support glob patterns.
    #[must_use]
    fn with_path(self, path: PathBuf) -> Self;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    fn rechunk(&self) -> bool;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(self, toggle: bool) -> Self;

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        None
    }

    /// Get list of files referenced by this reader.
    ///
    /// Returns [None] if path is not a glob pattern.
    fn glob(&self) -> PolarsResult<Option<GlobIterator>> {
        let path_str = self.path().to_string_lossy();
        if path_str.contains('*') {
            polars_glob(&path_str, self.cloud_options()).map(Some)
        } else {
            Ok(None)
        }
    }
}
