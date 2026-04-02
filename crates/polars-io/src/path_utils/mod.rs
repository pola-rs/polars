use std::borrow::Cow;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use polars_buffer::Buffer;
use polars_core::config;
use polars_core::error::{PolarsResult, polars_bail, to_compute_err};
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;

#[cfg(feature = "cloud")]
mod hugging_face;

use crate::cloud::CloudOptions;

#[allow(clippy::bind_instead_of_map)]
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
                std::env::temp_dir().join(format!("polars-{v}/"))
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
                        "permission mismatch: {perms:?}"
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
                "error initializing temporary directory: {e} \
                 consider explicitly setting POLARS_TEMP_DIR"
            ),
        )
    })
    .unwrap()
});

/// Replaces a "~" in the Path with the home directory.
pub fn resolve_homedir<'a, S: AsRef<Path> + ?Sized>(path: &'a S) -> Cow<'a, Path> {
    return inner(path.as_ref());

    fn inner(path: &Path) -> Cow<'_, Path> {
        if path.starts_with("~") {
            // home crate does not compile on wasm https://github.com/rust-lang/cargo/issues/12297
            #[cfg(not(target_family = "wasm"))]
            if let Some(homedir) = home::home_dir() {
                return Cow::Owned(homedir.join(path.strip_prefix("~").unwrap()));
            }
        }

        Cow::Borrowed(path)
    }
}

fn has_glob(path: &[u8]) -> bool {
    return get_glob_start_idx(path).is_some();

    /// Get the index of the first occurrence of a glob symbol.
    fn get_glob_start_idx(path: &[u8]) -> Option<usize> {
        memchr::memchr3(b'*', b'?', b'[', path)
    }
}

/// Returns `true` if `expanded_paths` were expanded from a single directory
pub fn expanded_from_single_directory(paths: &[PlRefPath], expanded_paths: &[PlRefPath]) -> bool {
    // Single input that isn't a glob
    paths.len() == 1 && !has_glob(paths[0].strip_scheme().as_bytes())
    // And isn't a file
    && {
        (
            // For local paths, we can just use `is_dir`
            !paths[0].has_scheme() && paths[0].as_std_path().is_dir()
        )
        || (
            // For cloud paths, we determine that the input path isn't a file by checking that the
            // output path differs.
            expanded_paths.is_empty() || (paths[0] != expanded_paths[0])
        )
    }
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
pub async fn expand_paths(
    paths: &[PlRefPath],
    glob: bool,
    hidden_file_prefix: &[PlSmallStr],
    #[allow(unused_variables)] cloud_options: &mut Option<CloudOptions>,
) -> PolarsResult<Buffer<PlRefPath>> {
    expand_paths_hive(paths, glob, hidden_file_prefix, cloud_options, false)
        .await
        .map(|x| x.0)
}

struct HiveIdxTracker<'a> {
    idx: usize,
    paths: &'a [PlRefPath],
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
                &paths[path_idx - 1],
                &paths[path_idx],
            )
        } else {
            self.idx = std::cmp::min(self.idx, i);
            Ok(())
        }
    }
}

#[cfg(feature = "cloud")]
async fn expand_path_cloud(
    path: PlRefPath,
    cloud_options: Option<&CloudOptions>,
    glob: bool,
    first_path_has_scheme: bool,
) -> PolarsResult<(usize, Vec<PlRefPath>)> {
    let format_path = |scheme: &str, bucket: &str, location: &str| {
        if first_path_has_scheme {
            format!("{scheme}://{bucket}/{location}")
        } else {
            format!("/{location}")
        }
    };

    use polars_utils::_limit_path_len_io_err;

    use crate::cloud::object_path_from_str;
    let path_str = path.as_str();

    let (cloud_location, store) =
        crate::cloud::build_object_store(path.clone(), cloud_options, glob).await?;
    let prefix = object_path_from_str(&cloud_location.prefix)?;

    let out = if !path_str.ends_with("/") && (!glob || cloud_location.expansion.is_none()) && {
        // We need to check if it is a directory for local paths (we can be here due
        // to FORCE_ASYNC). For cloud paths the convention is that the user must add
        // a trailing slash `/` to scan directories. We don't infer it as that would
        // mean sending one network request per path serially (very slow).
        path.has_scheme() || path.as_std_path().is_file()
    } {
        (
            0,
            vec![PlRefPath::new(format_path(
                cloud_location.scheme,
                &cloud_location.bucket,
                prefix.as_ref(),
            ))],
        )
    } else {
        use futures::TryStreamExt;

        if !path.has_scheme() {
            // FORCE_ASYNC in the test suite wants us to raise a proper error message
            // for non-existent file paths. Note we can't do this for cloud paths as
            // there is no concept of a "directory" - a non-existent path is
            // indistinguishable from an empty directory.
            path.as_std_path()
                .metadata()
                .map_err(|err| _limit_path_len_io_err(path.as_std_path(), err))?;
        }

        let cloud_location = &cloud_location;
        let prefix_ref = &prefix;

        let mut paths = store
            .exec_with_rebuild_retry_on_err(|s| async move {
                let out = s
                    .list(Some(prefix_ref))
                    .try_filter_map(|x| async move {
                        let out = (x.size > 0).then(|| {
                            PlRefPath::new({
                                format_path(
                                    cloud_location.scheme,
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
            })
            .await?;

        // Since Path::parse() removes any trailing slash ('/'), we may need to restore it
        // to calculate the right byte offset
        let mut prefix = prefix.to_string();
        if path_str.ends_with('/') && !prefix.ends_with('/') {
            prefix.push('/')
        };

        paths.sort_unstable();

        (
            format_path(
                cloud_location.scheme,
                &cloud_location.bucket,
                prefix.as_ref(),
            )
            .len(),
            paths,
        )
    };

    PolarsResult::Ok(out)
}

/// Recursively traverses directories and expands globs if `glob` is `true`.
/// Returns the expanded paths and the index at which to start parsing hive
/// partitions from the path.
pub async fn expand_paths_hive(
    paths: &[PlRefPath],
    glob: bool,
    hidden_file_prefix: &[PlSmallStr],
    #[allow(unused_variables)] cloud_options: &mut Option<CloudOptions>,
    check_directory_level: bool,
) -> PolarsResult<(Buffer<PlRefPath>, usize)> {
    let Some(first_path) = paths.first() else {
        return Ok((vec![].into(), 0));
    };

    let first_path_has_scheme = first_path.has_scheme();

    let is_hidden_file = move |path: &PlRefPath| {
        path.file_name()
            .and_then(|x| x.to_str())
            .is_some_and(|file_name| {
                hidden_file_prefix
                    .iter()
                    .any(|x| file_name.starts_with(x.as_str()))
            })
    };

    let mut out_paths = OutPaths {
        paths: vec![],
        exts: [None, None],
        is_hidden_file: &is_hidden_file,
    };

    let mut hive_idx_tracker = HiveIdxTracker {
        idx: usize::MAX,
        paths,
        check_directory_level,
    };

    if first_path_has_scheme || {
        cfg!(not(target_family = "windows")) && polars_config::config().force_async()
    } {
        #[cfg(feature = "cloud")]
        {
            if first_path.scheme() == Some(CloudScheme::Hf) {
                let (expand_start_idx, paths) = hugging_face::expand_paths_hf(
                    paths,
                    check_directory_level,
                    cloud_options,
                    glob,
                )
                .await?;

                return Ok((paths.into(), expand_start_idx));
            }

            for (path_idx, path) in paths.iter().enumerate() {
                use std::borrow::Cow;

                let mut path = Cow::Borrowed(path);

                if matches!(path.scheme(), Some(CloudScheme::Http | CloudScheme::Https)) {
                    let mut rewrite_aws = false;

                    #[cfg(feature = "aws")]
                    if let Some(p) = (|| {
                        use crate::cloud::CloudConfig;

                        // See https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html#virtual-hosted-style-access
                        // Path format: https://bucket-name.s3.region-code.amazonaws.com/key-name
                        let after_scheme = path.strip_scheme();

                        let bucket_end = after_scheme.find(".s3.")?;
                        let offset = bucket_end + 4;
                        // Search after offset to prevent matching `.s3.amazonaws.com` (legacy global endpoint URL without region).
                        let region_end = offset + after_scheme[offset..].find(".amazonaws.com/")?;

                        // Do not convert if '?' (this can be query parameters for AWS presigned URLs).
                        if after_scheme[..region_end].contains('/') || after_scheme.contains('?') {
                            return None;
                        }

                        let bucket = &after_scheme[..bucket_end];
                        let region = &after_scheme[bucket_end + 4..region_end];
                        let key = &after_scheme[region_end + 15..];

                        if let CloudConfig::Aws(configs) = cloud_options
                            .get_or_insert_default()
                            .config
                            .get_or_insert_with(|| CloudConfig::Aws(Vec::with_capacity(1)))
                        {
                            use object_store::aws::AmazonS3ConfigKey;

                            if !matches!(configs.last(), Some((AmazonS3ConfigKey::Region, _))) {
                                configs.push((AmazonS3ConfigKey::Region, region.into()))
                            }
                        }

                        Some(format!("s3://{bucket}/{key}"))
                    })() {
                        path = Cow::Owned(PlRefPath::new(p));
                        rewrite_aws = true;
                    }

                    if !rewrite_aws {
                        out_paths.push(path.into_owned());
                        hive_idx_tracker.update(0, path_idx)?;
                        continue;
                    }
                }

                let sort_start_idx = out_paths.paths.len();

                if glob && has_glob(path.as_bytes()) {
                    hive_idx_tracker.update(0, path_idx)?;

                    let iter = crate::pl_async::get_runtime().block_in_place_on(
                        crate::async_glob(path.into_owned(), cloud_options.as_ref()),
                    )?;

                    if first_path_has_scheme {
                        out_paths.extend(iter.into_iter().map(PlRefPath::new))
                    } else {
                        // FORCE_ASYNC, remove leading file:// as the caller may not be expecting a
                        // URI result.
                        out_paths.extend(iter.iter().map(|x| &x[7..]).map(PlRefPath::new))
                    };
                } else {
                    let (expand_start_idx, paths) = expand_path_cloud(
                        path.into_owned(),
                        cloud_options.as_ref(),
                        glob,
                        first_path_has_scheme,
                    )
                    .await?;
                    out_paths.extend_from_slice(&paths);
                    hive_idx_tracker.update(expand_start_idx, path_idx)?;
                };

                if let Some(mut_slice) = out_paths.paths.get_mut(sort_start_idx..) {
                    <[PlRefPath]>::sort_unstable(mut_slice);
                }
            }
        }
        #[cfg(not(feature = "cloud"))]
        panic!("Feature `cloud` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let mut stack: VecDeque<Cow<'_, Path>> = VecDeque::new();
        let mut paths_scratch: Vec<PathBuf> = vec![];

        for (path_idx, path) in paths.iter().enumerate() {
            stack.clear();
            let sort_start_idx = out_paths.paths.len();

            if path.as_std_path().is_dir() {
                let i = path.as_str().len();

                hive_idx_tracker.update(i, path_idx)?;

                stack.push_back(Cow::Borrowed(path.as_std_path()));

                while let Some(dir) = stack.pop_front() {
                    let mut last_err = Ok(());

                    paths_scratch.clear();
                    paths_scratch.extend(std::fs::read_dir(dir)?.map_while(|x| {
                        match x.map(|x| x.path()) {
                            Ok(v) => Some(v),
                            Err(e) => {
                                last_err = Err(e);
                                None
                            },
                        }
                    }));

                    last_err?;

                    for path in paths_scratch.drain(..) {
                        let md = path.metadata()?;

                        if md.is_dir() {
                            stack.push_back(Cow::Owned(path));
                        } else if md.len() > 0 {
                            out_paths.push(PlRefPath::try_from_path(&path)?);
                        }
                    }
                }
            } else if glob && has_glob(path.as_bytes()) {
                hive_idx_tracker.update(0, path_idx)?;

                let Ok(paths) = glob::glob(path.as_str()) else {
                    polars_bail!(ComputeError: "invalid glob pattern given")
                };

                for path in paths {
                    let path = path.map_err(to_compute_err)?;
                    let md = path.metadata()?;
                    if !md.is_dir() && md.len() > 0 {
                        out_paths.push(PlRefPath::try_from_path(&path)?);
                    }
                }
            } else {
                hive_idx_tracker.update(0, path_idx)?;
                out_paths.push(path.clone());
            };

            if let Some(mut_slice) = out_paths.paths.get_mut(sort_start_idx..) {
                <[PlRefPath]>::sort_unstable(mut_slice);
            }
        }
    }

    if expanded_from_single_directory(paths, out_paths.paths.as_slice()) {
        if let [Some((_, p1)), Some((_, p2))] = out_paths.exts {
            polars_bail!(
                InvalidOperation: "directory contained paths with different file extensions: \
                first path: {}, second path: {}. Please use a glob pattern to explicitly specify \
                which files to read (e.g. 'dir/**/*', 'dir/**/*.parquet')",
                &p1, &p2
            )
        }
    }

    return Ok((out_paths.paths.into(), hive_idx_tracker.idx));

    /// Wrapper around `Vec<PathBuf>` that also tracks file extensions, so that
    /// we don't have to traverse the entire list again to validate extensions.
    struct OutPaths<'a, F: Fn(&PlRefPath) -> bool> {
        paths: Vec<PlRefPath>,
        exts: [Option<(PlSmallStr, PlRefPath)>; 2],
        is_hidden_file: &'a F,
    }

    impl<F> OutPaths<'_, F>
    where
        F: Fn(&PlRefPath) -> bool,
    {
        fn push(&mut self, value: PlRefPath) {
            if (self.is_hidden_file)(&value) {
                return;
            }

            let exts = &mut self.exts;
            Self::update_ext_status(exts, &value);

            self.paths.push(value)
        }

        fn extend(&mut self, values: impl IntoIterator<Item = PlRefPath>) {
            let exts = &mut self.exts;

            self.paths.extend(
                values
                    .into_iter()
                    .filter(|x| !(self.is_hidden_file)(x))
                    .inspect(|x| {
                        Self::update_ext_status(exts, x);
                    }),
            )
        }

        fn extend_from_slice(&mut self, values: &[PlRefPath]) {
            self.extend(values.iter().cloned())
        }

        fn update_ext_status(exts: &mut [Option<(PlSmallStr, PlRefPath)>; 2], value: &PlRefPath) {
            let ext = value
                .extension()
                .map_or(PlSmallStr::EMPTY, PlSmallStr::from);

            if exts[0].is_none() {
                exts[0] = Some((ext, value.clone()));
            } else if exts[1].is_none() && ext != exts[0].as_ref().unwrap().0 {
                exts[1] = Some((ext, value.clone()));
            }
        }
    }
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

    use polars_utils::pl_path::PlRefPath;

    use super::resolve_homedir;
    use crate::pl_async::get_runtime;

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

        let resolved: Vec<PathBuf> = paths
            .iter()
            .map(resolve_homedir)
            .map(|x| x.into_owned())
            .collect();

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

        let resolved: Vec<PathBuf> = paths
            .iter()
            .map(resolve_homedir)
            .map(|x| x.into_owned())
            .collect();

        assert_eq!(resolved[0], paths[0]);
        assert_eq!(resolved[1].file_name(), paths[1].file_name());
        assert!(resolved[1].is_absolute());
        assert!(resolved[2].is_absolute());
    }

    #[test]
    fn test_http_path_with_query_parameters_is_not_expanded_as_glob() {
        // Don't confuse HTTP URL's with query parameters for globs.
        // See https://github.com/pola-rs/polars/pull/17774

        use super::expand_paths;

        let path = "https://pola.rs/test.csv?token=bear";
        let paths = &[PlRefPath::new(path)];
        let out = get_runtime()
            .block_on(expand_paths(paths, true, &[], &mut None))
            .unwrap();
        assert_eq!(out.as_ref(), paths);
    }
}
