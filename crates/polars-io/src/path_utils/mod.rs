use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use once_cell::sync::Lazy;
use polars_core::config;
use polars_core::error::{polars_bail, to_compute_err, PolarsError, PolarsResult};
use regex::Regex;

#[cfg(feature = "cloud")]
mod hugging_face;

use crate::cloud::CloudOptions;

pub static POLARS_TEMP_DIR_BASE_PATH: Lazy<Box<Path>> = Lazy::new(|| {
    let path = std::env::var("POLARS_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::temp_dir().to_string_lossy().as_ref()).join("polars/")
        })
        .into_boxed_path();

    if let Err(err) = std::fs::create_dir_all(path.as_ref()) {
        if !path.is_dir() {
            panic!(
                "failed to create temporary directory: path = {}, err = {}",
                path.to_str().unwrap(),
                err
            );
        }
    }

    path
});

/// Replaces a "~" in the Path with the home directory.
pub fn resolve_homedir(path: &Path) -> PathBuf {
    if path.starts_with("~") {
        // home crate does not compile on wasm https://github.com/rust-lang/cargo/issues/12297
        #[cfg(not(target_family = "wasm"))]
        if let Some(homedir) = home::home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}

static CLOUD_URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?|hf)://").unwrap());

/// Check if the path is a cloud url.
pub fn is_cloud_url<P: AsRef<Path>>(p: P) -> bool {
    match p.as_ref().as_os_str().to_str() {
        Some(s) => CLOUD_URL.is_match(s),
        _ => false,
    }
}

/// Get the index of the first occurrence of a glob symbol.
pub fn get_glob_start_idx(path: &[u8]) -> Option<usize> {
    memchr::memchr3(b'*', b'?', b'[', path)
}

/// Returns `true` if `expanded_paths` were expanded from a single directory
pub fn expanded_from_single_directory<P: AsRef<std::path::Path>>(
    paths: &[P],
    expanded_paths: &[P],
) -> bool {
    // Single input that isn't a glob
    paths.len() == 1 && get_glob_start_idx(paths[0].as_ref().to_str().unwrap().as_bytes()).is_none()
    // And isn't a file
    && {
        (
            // For local paths, we can just use `is_dir`
            !is_cloud_url(paths[0].as_ref()) && paths[0].as_ref().is_dir()
        )
        || (
            // Otherwise we check the output path is different from the input path, so that we also
            // handle the case of a directory containing a single file.
            !expanded_paths.is_empty() && (paths[0].as_ref() != expanded_paths[0].as_ref())
        )
    }
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
pub fn expand_paths(
    paths: &[PathBuf],
    glob: bool,
    #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Arc<Vec<PathBuf>>> {
    expand_paths_hive(paths, glob, cloud_options, false).map(|x| x.0)
}

struct HiveIdxTracker<'a> {
    idx: usize,
    paths: &'a [PathBuf],
    check_directory_level: bool,
}

impl<'a> HiveIdxTracker<'a> {
    fn update(&mut self, i: usize, path_idx: usize) -> PolarsResult<()> {
        let check_directory_level = self.check_directory_level;
        let paths = self.paths;

        if check_directory_level
            && ![usize::MAX, i].contains(&self.idx)
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
            self.idx = std::cmp::min(self.idx, i);
            Ok(())
        }
    }
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
/// Returns the expanded paths and the index at which to start parsing hive
/// partitions from the path.
pub fn expand_paths_hive(
    paths: &[PathBuf],
    glob: bool,
    #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    check_directory_level: bool,
) -> PolarsResult<(Arc<Vec<PathBuf>>, usize)> {
    let Some(first_path) = paths.first() else {
        return Ok((vec![].into(), 0));
    };

    let is_cloud = is_cloud_url(first_path);
    let mut out_paths = vec![];

    let mut hive_idx_tracker = HiveIdxTracker {
        idx: usize::MAX,
        paths,
        check_directory_level,
    };

    if is_cloud || { cfg!(not(target_family = "windows")) && config::force_async() } {
        #[cfg(feature = "cloud")]
        {
            use polars_utils::_limit_path_len_io_err;

            use crate::cloud::object_path_from_str;

            if first_path.starts_with("hf://") {
                let (expand_start_idx, paths) = crate::pl_async::get_runtime()
                    .block_on_potential_spawn(hugging_face::expand_paths_hf(
                        paths,
                        check_directory_level,
                        cloud_options,
                        glob,
                    ))?;

                return Ok((Arc::from(paths), expand_start_idx));
            }

            let format_path = |scheme: &str, bucket: &str, location: &str| {
                if is_cloud {
                    format!("{}://{}/{}", scheme, bucket, location)
                } else {
                    format!("/{}", location)
                }
            };

            let expand_path_cloud = |path: &str,
                                     cloud_options: Option<&CloudOptions>|
             -> PolarsResult<(usize, Vec<PathBuf>)> {
                crate::pl_async::get_runtime().block_on_potential_spawn(async {
                    let (cloud_location, store) =
                        crate::cloud::build_object_store(path, cloud_options, glob).await?;
                    let prefix = object_path_from_str(&cloud_location.prefix)?;

                    let out = if !path.ends_with("/")
                        && (!glob || cloud_location.expansion.is_none())
                        && {
                            // We need to check if it is a directory for local paths (we can be here due
                            // to FORCE_ASYNC). For cloud paths the convention is that the user must add
                            // a trailing slash `/` to scan directories. We don't infer it as that would
                            // mean sending one network request per path serially (very slow).
                            is_cloud || PathBuf::from(path).is_file()
                        } {
                        (
                            0,
                            vec![PathBuf::from(format_path(
                                &cloud_location.scheme,
                                &cloud_location.bucket,
                                prefix.as_ref(),
                            ))],
                        )
                    } else {
                        use futures::TryStreamExt;

                        if !is_cloud {
                            // FORCE_ASYNC in the test suite wants us to raise a proper error message
                            // for non-existent file paths. Note we can't do this for cloud paths as
                            // there is no concept of a "directory" - a non-existent path is
                            // indistinguishable from an empty directory.
                            let path = PathBuf::from(path);
                            if !path.is_dir() {
                                path.metadata()
                                    .map_err(|err| _limit_path_len_io_err(&path, err))?;
                            }
                        }

                        let cloud_location = &cloud_location;

                        let mut paths = store
                            .list(Some(&prefix))
                            .try_filter_map(|x| async move {
                                let out = (x.size > 0).then(|| {
                                    PathBuf::from({
                                        format_path(
                                            &cloud_location.scheme,
                                            &cloud_location.bucket,
                                            x.location.as_ref(),
                                        )
                                    })
                                });
                                Ok(out)
                            })
                            .try_collect::<Vec<_>>()
                            .await
                            .map_err(to_compute_err)?;

                        paths.sort_unstable();
                        (
                            format_path(
                                &cloud_location.scheme,
                                &cloud_location.bucket,
                                &cloud_location.prefix,
                            )
                            .len(),
                            paths,
                        )
                    };

                    PolarsResult::Ok(out)
                })
            };

            for (path_idx, path) in paths.iter().enumerate() {
                if path.to_str().unwrap().starts_with("http") {
                    out_paths.push(path.clone());
                    hive_idx_tracker.update(0, path_idx)?;
                    continue;
                }

                let glob_start_idx = get_glob_start_idx(path.to_str().unwrap().as_bytes());

                let path = if glob && glob_start_idx.is_some() {
                    path.clone()
                } else {
                    let (expand_start_idx, paths) =
                        expand_path_cloud(path.to_str().unwrap(), cloud_options)?;
                    out_paths.extend_from_slice(&paths);
                    hive_idx_tracker.update(expand_start_idx, path_idx)?;
                    continue;
                };

                hive_idx_tracker.update(0, path_idx)?;

                let iter = crate::pl_async::get_runtime().block_on_potential_spawn(
                    crate::async_glob(path.to_str().unwrap(), cloud_options),
                )?;

                if is_cloud {
                    out_paths.extend(iter.into_iter().map(PathBuf::from));
                } else {
                    // FORCE_ASYNC, remove leading file:// as not all readers support it.
                    out_paths.extend(iter.iter().map(|x| &x[7..]).map(PathBuf::from))
                }
            }
        }
        #[cfg(not(feature = "cloud"))]
        panic!("Feature `cloud` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let mut stack = VecDeque::new();

        for path_idx in 0..paths.len() {
            let path = &paths[path_idx];
            stack.clear();

            if path.is_dir() {
                let i = path.to_str().unwrap().len();

                hive_idx_tracker.update(i, path_idx)?;

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
                        } else if path.metadata()?.len() > 0 {
                            out_paths.push(path);
                        }
                    }
                }

                continue;
            }

            let i = get_glob_start_idx(path.to_str().unwrap().as_bytes());

            if glob && i.is_some() {
                hive_idx_tracker.update(0, path_idx)?;

                let Ok(paths) = glob::glob(path.to_str().unwrap()) else {
                    polars_bail!(ComputeError: "invalid glob pattern given")
                };

                for path in paths {
                    let path = path.map_err(to_compute_err)?;
                    if !path.is_dir() && path.metadata()?.len() > 0 {
                        out_paths.push(path);
                    }
                }
            } else {
                hive_idx_tracker.update(0, path_idx)?;
                out_paths.push(path.clone());
            }
        }
    }

    let out_paths = if expanded_from_single_directory(paths, out_paths.as_ref()) {
        // Require all file extensions to be the same when expanding a single directory.
        let ext = out_paths[0].extension();

        (0..out_paths.len())
            .map(|i| {
                let path = out_paths[i].clone();

                if path.extension() != ext {
                    polars_bail!(
                        InvalidOperation: r#"directory contained paths with different file extensions: \
                        first path: {}, second path: {}. Please use a glob pattern to explicitly specify \
                        which files to read (e.g. "dir/**/*", "dir/**/*.parquet")"#,
                        out_paths[i - 1].to_str().unwrap(), path.to_str().unwrap()
                    );
                };

                Ok(path)
            })
            .collect::<PolarsResult<Vec<_>>>()?
    } else {
        out_paths
    };

    Ok((Arc::new(out_paths), hive_idx_tracker.idx))
}

/// Ignores errors from `std::fs::create_dir_all` if the directory exists.
#[cfg(feature = "file_cache")]
pub(crate) fn ensure_directory_init(path: &Path) -> std::io::Result<()> {
    let result = std::fs::create_dir_all(path);

    if path.is_dir() {
        Ok(())
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::resolve_homedir;

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_resolve_homedir() {
        let paths: Vec<PathBuf> = vec![
            "~/dir1/dir2/test.csv".into(),
            "/abs/path/test.csv".into(),
            "rel/path/test.csv".into(),
            "/".into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0].file_name(), paths[0].file_name());
        assert!(resolved[0].is_absolute());
        assert_eq!(resolved[1], paths[1]);
        assert_eq!(resolved[2], paths[2]);
        assert_eq!(resolved[3], paths[3]);
        assert!(resolved[4].is_absolute());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_resolve_homedir_windows() {
        let paths: Vec<PathBuf> = vec![
            r#"c:\Users\user1\test.csv"#.into(),
            r#"~\user1\test.csv"#.into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0], paths[0]);
        assert_eq!(resolved[1].file_name(), paths[1].file_name());
        assert!(resolved[1].is_absolute());
        assert!(resolved[2].is_absolute());
    }

    #[test]
    fn test_http_path_with_query_parameters_is_not_expanded_as_glob() {
        // Don't confuse HTTP URL's with query parameters for globs.
        // See https://github.com/pola-rs/polars/pull/17774
        use std::path::PathBuf;

        use super::expand_paths;

        let path = "https://pola.rs/test.csv?token=bear";
        let paths = &[PathBuf::from(path)];
        let out = expand_paths(paths, true, None).unwrap();
        assert_eq!(out.as_ref(), paths);
    }
}
