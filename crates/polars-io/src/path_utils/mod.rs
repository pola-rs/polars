use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use polars_core::config;
use polars_core::error::{PolarsError, PolarsResult, polars_bail, to_compute_err};
use polars_utils::pl_str::PlSmallStr;

#[cfg(feature = "cloud")]
mod hugging_face;

use crate::cloud::CloudOptions;

pub static POLARS_TEMP_DIR_BASE_PATH: LazyLock<Box<Path>> = LazyLock::new(|| {
    (|| {
        let verbose = config::verbose();

        let path = if let Ok(v) = std::env::var("POLARS_TEMP_DIR").map(PathBuf::from) {
            if verbose {
                eprintln!("init_temp_dir: sourced from POLARS_TEMP_DIR")
            }
            v
        } else if cfg!(target_family = "unix") {
            let id = std::env::var("USER")
                .inspect(|_| {
                    if verbose {
                        eprintln!("init_temp_dir: sourced $USER")
                    }
                })
                .or_else(|_e| {
                    // We shouldn't hit here, but we can fallback to hashing $HOME if blake3 is
                    // available (it is available when file_cache is activated).
                    #[cfg(feature = "file_cache")]
                    {
                        std::env::var("HOME")
                            .inspect(|_| {
                                if verbose {
                                    eprintln!("init_temp_dir: sourced $HOME")
                                }
                            })
                            .map(|x| blake3::hash(x.as_bytes()).to_hex()[..32].to_string())
                    }
                    #[cfg(not(feature = "file_cache"))]
                    {
                        Err(_e)
                    }
                });

            if let Ok(v) = id {
                std::env::temp_dir().join(format!("polars-{}/", v))
            } else {
                return Err(std::io::Error::other(
                    "could not load $USER or $HOME environment variables",
                ));
            }
        } else if cfg!(target_family = "windows") {
            // Setting permissions on Windows is not as easy compared to Unix, but fortunately
            // the default temporary directory location is underneath the user profile, so we
            // shouldn't need to do anything.
            std::env::temp_dir().join("polars/")
        } else {
            std::env::temp_dir().join("polars/")
        }
        .into_boxed_path();

        if let Err(err) = std::fs::create_dir_all(path.as_ref()) {
            if !path.is_dir() {
                panic!(
                    "failed to create temporary directory: {} (path = {:?})",
                    err,
                    path.as_ref()
                );
            }
        }

        #[cfg(target_family = "unix")]
        {
            use std::os::unix::fs::PermissionsExt;

            let result = (|| {
                std::fs::set_permissions(path.as_ref(), std::fs::Permissions::from_mode(0o700))?;
                let perms = std::fs::metadata(path.as_ref())?.permissions();

                if (perms.mode() % 0o1000) != 0o700 {
                    std::io::Result::Err(std::io::Error::other(format!(
                        "permission mismatch: {:?}",
                        perms
                    )))
                } else {
                    std::io::Result::Ok(())
                }
            })()
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "error setting temporary directory permissions: {} (path = {:?})",
                        e,
                        path.as_ref()
                    ),
                )
            });

            if std::env::var("POLARS_ALLOW_UNSECURED_TEMP_DIR").as_deref() != Ok("1") {
                result?;
            }
        }

        std::io::Result::Ok(path)
    })()
    .map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "error initializing temporary directory: {} \
                 consider explicitly setting POLARS_TEMP_DIR",
                e
            ),
        )
    })
    .unwrap()
});

/// Replaces a "~" in the Path with the home directory.
pub fn resolve_homedir(path: &dyn AsRef<Path>) -> PathBuf {
    let path = path.as_ref();

    if path.starts_with("~") {
        // home crate does not compile on wasm https://github.com/rust-lang/cargo/issues/12297
        #[cfg(not(target_family = "wasm"))]
        if let Some(homedir) = home::home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}

polars_utils::regex_cache::cached_regex! {
    static CLOUD_URL = r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?|hf)://";
}

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
            // For cloud paths, we determine that the input path isn't a file by checking that the
            // output path differs.
            expanded_paths.is_empty() || (paths[0].as_ref() != expanded_paths[0].as_ref())
        )
    }
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
pub fn expand_paths(
    paths: &[PathBuf],
    glob: bool,
    #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Arc<[PathBuf]>> {
    expand_paths_hive(paths, glob, cloud_options, false).map(|x| x.0)
}

struct HiveIdxTracker<'a> {
    idx: usize,
    paths: &'a [PathBuf],
    check_directory_level: bool,
}

impl HiveIdxTracker<'_> {
    fn update(&mut self, i: usize, path_idx: usize) -> PolarsResult<()> {
        let check_directory_level = self.check_directory_level;
        let paths = self.paths;

        if check_directory_level
            && ![usize::MAX, i].contains(&self.idx)
            // They could still be the same directory level, just with different name length
            && (path_idx > 0 && paths[path_idx].parent() != paths[path_idx - 1].parent())
        {
            polars_bail!(
                InvalidOperation:
                "attempted to read from different directory levels with hive partitioning enabled: \
                first path: {}, second path: {}",
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
) -> PolarsResult<(Arc<[PathBuf]>, usize)> {
    let Some(first_path) = paths.first() else {
        return Ok((vec![].into(), 0));
    };

    let is_cloud = is_cloud_url(first_path);

    /// Wrapper around `Vec<PathBuf>` that also tracks file extensions, so that
    /// we don't have to traverse the entire list again to validate extensions.
    struct OutPaths {
        paths: Vec<PathBuf>,
        exts: [Option<(PlSmallStr, usize)>; 2],
        current_idx: usize,
    }

    impl OutPaths {
        fn update_ext_status(
            current_idx: &mut usize,
            exts: &mut [Option<(PlSmallStr, usize)>; 2],
            value: &Path,
        ) {
            let ext = value
                .extension()
                .map(|x| PlSmallStr::from(x.to_str().unwrap()))
                .unwrap_or(PlSmallStr::EMPTY);

            if exts[0].is_none() {
                exts[0] = Some((ext, *current_idx));
            } else if exts[1].is_none() && ext != exts[0].as_ref().unwrap().0 {
                exts[1] = Some((ext, *current_idx));
            }

            *current_idx += 1;
        }

        fn push(&mut self, value: PathBuf) {
            {
                let current_idx = &mut self.current_idx;
                let exts = &mut self.exts;
                Self::update_ext_status(current_idx, exts, &value);
            }
            self.paths.push(value)
        }

        fn extend(&mut self, values: impl IntoIterator<Item = PathBuf>) {
            let current_idx = &mut self.current_idx;
            let exts = &mut self.exts;

            self.paths.extend(values.into_iter().inspect(|x| {
                Self::update_ext_status(current_idx, exts, x);
            }))
        }

        fn extend_from_slice(&mut self, values: &[PathBuf]) {
            self.extend(values.iter().cloned())
        }
    }

    let mut out_paths = OutPaths {
        paths: vec![],
        exts: [None, None],
        current_idx: 0,
    };

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
                let (expand_start_idx, paths) = crate::pl_async::get_runtime().block_in_place_on(
                    hugging_face::expand_paths_hf(
                        paths,
                        check_directory_level,
                        cloud_options,
                        glob,
                    ),
                )?;

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
                crate::pl_async::get_runtime().block_in_place_on(async {
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
                            .try_exec_rebuild_on_err(|store| {
                                let st = store.clone();

                                async {
                                    let store = st;
                                    let out = store
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
                                        .await?;

                                    Ok(out)
                                }
                            })
                            .await?;

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

                let iter = crate::pl_async::get_runtime()
                    .block_in_place_on(crate::async_glob(path.to_str().unwrap(), cloud_options))?;

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

    assert_eq!(out_paths.current_idx, out_paths.paths.len());

    if expanded_from_single_directory(paths, out_paths.paths.as_slice()) {
        if let [Some((_, i1)), Some((_, i2))] = out_paths.exts {
            polars_bail!(
                InvalidOperation: r#"directory contained paths with different file extensions: \
                first path: {}, second path: {}. Please use a glob pattern to explicitly specify \
                which files to read (e.g. "dir/**/*", "dir/**/*.parquet")"#,
                &out_paths.paths[i1].to_string_lossy(), &out_paths.paths[i2].to_string_lossy()
            )
        }
    }

    Ok((out_paths.paths.into(), hive_idx_tracker.idx))
}

/// Ignores errors from `std::fs::create_dir_all` if the directory exists.
#[cfg(feature = "file_cache")]
pub(crate) fn ensure_directory_init(path: &Path) -> std::io::Result<()> {
    let result = std::fs::create_dir_all(path);

    if path.is_dir() { Ok(()) } else { result }
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
