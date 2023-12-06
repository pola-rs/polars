use std::fmt::Display;
use std::fs::File;
use std::path::{Path, PathBuf};

use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::mmap::{MmapBytesReader, ScanLocation};
use polars_io::{is_cloud_url, RowCount};

use crate::prelude::*;

pub type ScanLocationIterator = Box<dyn Iterator<Item = PolarsResult<ScanLocation>>>;

// cloud_options is used only with async feature
#[allow(unused_variables)]
fn polars_glob(
    pattern: &str,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<ScanLocationIterator> {
    if is_cloud_url(pattern) {
        #[cfg(feature = "async")]
        {
            let paths = polars_io::async_glob(pattern, cloud_options)?;
            Ok(Box::new(
                paths
                    .into_iter()
                    .map(|a| Ok(ScanLocation::Remote { uri: a })),
            ))
        }
        #[cfg(not(feature = "async"))]
        panic!("Feature `async` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let paths = glob::glob(pattern)
            .map_err(|_| polars_err!(ComputeError: "invalid glob pattern given"))?;
        let paths = paths.map(|v| {
            v.map_err(to_compute_err).and_then(|p| {
                File::open(p)
                    .map(|f| ScanLocation::Local {
                        path: p,
                        reader: Box::new(f),
                    })
                    .map_err(to_compute_err)
            })
        });
        Ok(Box::new(paths))
    }
}

/// Convert a more convenient input into an iterator over [ScanLocation].
pub trait TryIntoScanLocations {
    // We can't use TryInto/TryFrom because of the need for cloud_options.
    fn try_into_scanlocations(
        &self,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<ScanLocationIterator>;
}

impl TryIntoScanLocations for &Path {
    fn try_into_scanlocations(&self, cloud_options: Option<&CloudOptions>) -> PolarsResult<Self> {
        let path_str = path.to_string_lossy();
        polars_glob(&path_str, cloud_options)
    }
}

impl<P: Into<PathBuf>> TryIntoScanLocations for Vec<P> {
    fn try_into_scanlocations(&self, cloud_options: Option<&CloudOptions>) -> PolarsResult<Self> {
        Ok(Box::new(paths.clone().into_iter().map(|path| {
            let path_str = path.to_string_lossy();
            if is_cloud_url(path_str.as_ref()) {
                Ok(ScanLocation::Remote {
                    location: path_str.to_string(),
                })
            } else {
                let f = File::open(path)?;
                Ok(ScanLocation::Local {
                    path,
                    reader: Box::new(f),
                })
            }
        })))
    }
}

/// Reads [LazyFrame] from a filesystem or a cloud storage.
/// Supports glob patterns.
///
/// Use [LazyFileListReader::load_multiple] or other related methods to get the
/// final [LazyFrame].
pub trait LazyFileListReader: Clone {
    /// Get the final [LazyFrame]. A [str], [Path], or [PathBuf] will be treated
    /// as a glob of either filesystem paths or cloud URLs. A [Vec] of these
    /// will be treated as a list of specific filesystem paths or cloud URLs.
    fn load_multiple<SL: TryIntoScanLocations>(
        self,
        scan_locations: SL,
    ) -> PolarsResult<LazyFrame> {
        let readers = scan_location.try_into_scanlocations(self.cloud_options())?;
        let lfs = readers
            .map(|r| {
                let r = r?;
                self.clone()
                    .with_rechunk(false)
                    .load_specific(r)
                    .map_err(|e| {
                        polars_err!(
                            ComputeError: "error while reading {}: {}", r, e
                        )
                    })
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        polars_ensure!(
            !lfs.is_empty(),
            ComputeError: "no matching files found in {}", self.multiple_readers()
        );

        let mut lf = self.concat_impl(lfs)?;
        if let Some(n_rows) = self.n_rows() {
            lf = lf.slice(0, n_rows as IdxSize)
        };
        if let Some(rc) = self.row_count() {
            lf = lf.with_row_count(&rc.name, Some(rc.offset))
        };

        Ok(lf)
    }

    /// Recommended concatenation of [LazyFrame]s from many input files.
    ///
    /// This method should not take into consideration [LazyFileListReader::n_rows]
    /// nor [LazyFileListReader::row_count].
    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        concat_impl(&lfs, self.rechunk(), true, true, false)
    }

    /// Get the final [LazyFrame] for a specific file on the filesystem.
    ///
    /// Unlike [LazyFileListReader::load_multiple], remote URLs will not be
    /// loaded, nor will globbing be done.
    fn load_specific_file<P: Into<PathBuf>>(self, path: P) -> PolarsResult<LazyFrame> {
        let path = path.into();
        self.load_specific(ScanLocation::LocalFile { path })
    }

    /// Get the final [LazyFrame].
    fn load_specific(self, reader: ScanLocation) -> PolarsResult<LazyFrame>;

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
}
