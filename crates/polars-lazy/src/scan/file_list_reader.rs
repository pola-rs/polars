use std::collections::VecDeque;
use std::path::PathBuf;

use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::utils::is_cloud_url;
use polars_io::RowIndex;
use polars_plan::prelude::UnionArgs;

use crate::prelude::*;

pub type PathIterator = Box<dyn Iterator<Item = PolarsResult<PathBuf>>>;

fn expand_paths(
    paths: &[PathBuf],
    #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    glob: bool,
) -> PolarsResult<(Arc<[PathBuf]>, usize)> {
    let Some(first_path) = paths.first() else {
        return Ok((vec![].into(), 0));
    };

    let is_cloud = is_cloud_url(first_path);
    let mut expand_start_idx = usize::MAX;
    let mut out_paths = vec![];

    if is_cloud {
        #[cfg(feature = "async")]
        {
            use polars_io::cloud::{CloudLocation, PolarsObjectStore};

            fn is_file_cloud(
                path: &str,
                cloud_options: Option<&CloudOptions>,
            ) -> PolarsResult<bool> {
                polars_io::pl_async::get_runtime().block_on_potential_spawn(async {
                    let (CloudLocation { prefix, .. }, store) =
                        polars_io::cloud::build_object_store(path, cloud_options).await?;
                    let store = PolarsObjectStore::new(store);
                    PolarsResult::Ok(store.head(&prefix.into()).await.is_ok())
                })
            }

            for path in paths {
                let glob_start_idx =
                    memchr::memchr3(b'*', b'?', b'[', path.to_str().unwrap().as_bytes());

                let (path, glob_start_idx) = if let Some(glob_start_idx) = glob_start_idx {
                    (path.clone(), glob_start_idx)
                } else if !path.ends_with("/")
                    && is_file_cloud(path.to_str().unwrap(), cloud_options)?
                {
                    out_paths.push(path.clone());
                    continue;
                } else if !glob {
                    polars_bail!(ComputeError: "not implemented: did not find cloud file at path = {} and `glob` was set to false", path.to_str().unwrap());
                } else {
                    (path.join("**/*"), path.to_str().unwrap().len())
                };

                expand_start_idx = std::cmp::min(expand_start_idx, glob_start_idx);

                out_paths.extend(
                    polars_io::async_glob(path.to_str().unwrap(), cloud_options)?
                        .into_iter()
                        .map(PathBuf::from),
                );
            }
        }
        #[cfg(not(feature = "async"))]
        panic!("Feature `async` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let mut stack = VecDeque::new();

        for path in paths {
            stack.clear();

            if path.is_dir() {
                let i = path.to_str().unwrap().len();

                expand_start_idx = std::cmp::min(expand_start_idx, i);

                stack.push_back(path.clone());

                while let Some(dir) = stack.pop_front() {
                    let mut paths = std::fs::read_dir(dir)
                        .map_err(PolarsError::from)?
                        .map(|x| x.map(|x| x.path()))
                        .collect::<std::io::Result<Vec<_>>>()
                        .map_err(PolarsError::from)?;
                    paths.sort_unstable();

                    for path in paths {
                        if path.is_dir() {
                            stack.push_back(path);
                        } else {
                            out_paths.push(path);
                        }
                    }
                }

                continue;
            }

            let i = memchr::memchr3(b'*', b'?', b'[', path.to_str().unwrap().as_bytes());

            if glob && i.is_some() {
                let i = i.unwrap();
                expand_start_idx = std::cmp::min(expand_start_idx, i);

                let Ok(paths) = glob::glob(path.to_str().unwrap()) else {
                    polars_bail!(ComputeError: "invalid glob pattern given")
                };

                for path in paths {
                    out_paths.push(path.map_err(to_compute_err)?);
                }
            } else {
                out_paths.push(path.clone());
            }
        }
    }

    // Todo:
    // This maintains existing behavior - will remove very soon.
    expand_start_idx = 0;

    Ok((
        out_paths.into_iter().collect::<Arc<[_]>>(),
        expand_start_idx,
    ))
}

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

        let paths = self.expand_paths()?.0;

        let lfs = paths
            .iter()
            .map(|path| {
                self.clone()
                    // Each individual reader should not apply a row limit.
                    .with_n_rows(None)
                    // Each individual reader should not apply a row index.
                    .with_row_index(None)
                    .with_paths(Arc::new([path.clone()]))
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
    fn with_paths(self, paths: Arc<[PathBuf]>) -> Self;

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

    /// Returns a list of paths after resolving globs and directories, as well as
    /// the hive start index.
    fn expand_paths(&self) -> PolarsResult<(Arc<[PathBuf]>, usize)> {
        expand_paths(self.paths(), self.cloud_options(), self.glob())
    }
}
