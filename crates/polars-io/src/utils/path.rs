use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use regex::Regex;

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
    Lazy::new(|| Regex::new(r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?)://").unwrap());

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
}
