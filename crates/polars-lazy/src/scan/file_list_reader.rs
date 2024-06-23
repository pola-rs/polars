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

pub(super) fn get_glob_start_idx(path: &[u8]) -> Option<usize> {
    memchr::memchr3(b'*', b'?', b'[', path)
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
/// Returns the expanded paths and the index at which to start parsing hive
/// partitions from the path.
fn expand_paths(
    paths: &[PathBuf],
    #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    glob: bool,
    check_directory_level: bool,
) -> PolarsResult<(Arc<[PathBuf]>, usize)> {
    let Some(first_path) = paths.first() else {
        return Ok((vec![].into(), 0));
    };

    let is_cloud = is_cloud_url(first_path);
    let mut out_paths = vec![];

    let expand_start_idx = &mut usize::MAX.clone();
    let mut update_expand_start_idx = |i, path_idx: usize| {
        if check_directory_level
            && ![usize::MAX, i].contains(expand_start_idx)
            // They could still be the same directory level, just with different name length
            && (paths[path_idx].parent() != paths[path_idx - 1].parent())
        {
            polars_bail!(
                InvalidOperation:
                "attempted to read from different directory levels with hive partitioning enabled: first path: {}, second path: {}",
                paths[path_idx - 1].to_str().unwrap(),
                paths[path_idx].to_str().unwrap(),
            )
        } else {
            *expand_start_idx = std::cmp::min(*expand_start_idx, i);
            Ok(())
        }
    };

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

            for (path_idx, path) in paths.iter().enumerate() {
                let glob_start_idx = get_glob_start_idx(path.to_str().unwrap().as_bytes());

                let path = if glob_start_idx.is_some() {
                    path.clone()
                } else if !path.ends_with("/")
                    && is_file_cloud(path.to_str().unwrap(), cloud_options)?
                {
                    update_expand_start_idx(0, path_idx)?;
                    out_paths.push(path.clone());
                    continue;
                } else if !glob {
                    polars_bail!(ComputeError: "not implemented: did not find cloud file at path = {} and `glob` was set to false", path.to_str().unwrap());
                } else {
                    // FIXME: This will fail! See https://github.com/pola-rs/polars/issues/17105
                    path.join("**/*")
                };

                update_expand_start_idx(0, path_idx)?;

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

        for path_idx in 0..paths.len() {
            let path = &paths[path_idx];
            stack.clear();

            if path.is_dir() {
                let i = path.to_str().unwrap().len();

                update_expand_start_idx(i, path_idx)?;

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

            let i = get_glob_start_idx(path.to_str().unwrap().as_bytes());

            if glob && i.is_some() {
                update_expand_start_idx(0, path_idx)?;

                let Ok(paths) = glob::glob(path.to_str().unwrap()) else {
                    polars_bail!(ComputeError: "invalid glob pattern given")
                };

                for path in paths {
                    out_paths.push(path.map_err(to_compute_err)?);
                }
            } else {
                update_expand_start_idx(0, path_idx)?;
                out_paths.push(path.clone());
            }
        }
    }

    Ok((
        out_paths.into_iter().collect::<Arc<[_]>>(),
        *expand_start_idx,
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

        let paths = self.expand_paths(false)?.0;

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
    /// the string index at which to start parsing hive partitions.
    fn expand_paths(&self, check_directory_level: bool) -> PolarsResult<(Arc<[PathBuf]>, usize)> {
        expand_paths(
            self.paths(),
            self.cloud_options(),
            self.glob(),
            check_directory_level,
        )
    }
}
