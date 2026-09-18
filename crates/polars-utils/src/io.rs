use std::fs::File;
use std::path::{Path, PathBuf};
use std::{fmt, io};

use polars_error::*;

/// An IO error together with the full path it occurred on.
///
/// The `Display` output truncates long paths so they stay readable in error messages, but
/// the full path remains available to consumers that downcast the payload of the
/// [`io::Error`] built by [`_limit_path_len_io_err`].
#[derive(Debug)]
pub struct PathIoError {
    pub path: PathBuf,
    pub source: io::Error,
}

impl PathIoError {
    const MAX_DISPLAYED_PATH_CHARS: usize = 88;
}

impl fmt::Display for PathIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.path.to_string_lossy();
        let n_chars = path.chars().count();
        if n_chars > Self::MAX_DISPLAYED_PATH_CHARS && !polars_config::config().verbose() {
            let truncated_path: String = path
                .chars()
                .skip(n_chars - Self::MAX_DISPLAYED_PATH_CHARS)
                .collect();
            write!(
                f,
                "{}: ...{truncated_path} (set POLARS_VERBOSE=1 to see full path)",
                self.source
            )
        } else {
            write!(f, "{}: {path}", self.source)
        }
    }
}

impl std::error::Error for PathIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Attaches `path` to `err`, keeping its [`io::ErrorKind`].
///
/// The path is available in full through [`PathIoError`], and truncated in the message.
pub fn _limit_path_len_io_err(path: &Path, err: io::Error) -> PolarsError {
    io::Error::new(
        err.kind(),
        PathIoError {
            path: path.to_path_buf(),
            source: err,
        },
    )
    .into()
}

pub fn open_file(path: &Path) -> PolarsResult<File> {
    File::open(path).map_err(|err| _limit_path_len_io_err(path, err))
}

pub fn open_file_write(path: &Path) -> PolarsResult<File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|err| _limit_path_len_io_err(path, err))
}

pub fn create_file(path: &Path) -> PolarsResult<File> {
    File::create(path).map_err(|err| _limit_path_len_io_err(path, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permission_denied() -> io::Error {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Permission denied (os error 13)",
        )
    }

    fn io_error_of(err: &PolarsError) -> &io::Error {
        match err {
            PolarsError::IO { error, .. } => error,
            other => panic!("expected an IO error, got {other:?}"),
        }
    }

    #[test]
    fn keeps_kind_and_full_path() {
        let path = PathBuf::from(format!("/{}/file.parquet", "a".repeat(200)));
        let err = _limit_path_len_io_err(&path, permission_denied());
        let io_err = io_error_of(&err);

        assert_eq!(io_err.kind(), io::ErrorKind::PermissionDenied);

        let with_path = io_err
            .get_ref()
            .and_then(|e| e.downcast_ref::<PathIoError>())
            .expect("payload should be a PathIoError");
        assert_eq!(with_path.path, path);
        assert_eq!(with_path.source.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn display_truncates_long_paths() {
        let path = PathBuf::from(format!("/{}/file.parquet", "a".repeat(200)));
        let msg = _limit_path_len_io_err(&path, permission_denied()).to_string();

        assert!(
            msg.starts_with("Permission denied (os error 13): ..."),
            "{msg}"
        );
        assert!(
            msg.ends_with("(set POLARS_VERBOSE=1 to see full path)"),
            "{msg}"
        );
        assert!(!msg.contains(path.to_str().unwrap()), "{msg}");
        assert!(msg.contains("aaa/file.parquet"), "{msg}");
    }

    #[test]
    fn display_keeps_short_paths() {
        let path = PathBuf::from("/tmp/out/file.parquet");
        let msg = _limit_path_len_io_err(&path, permission_denied()).to_string();
        assert_eq!(
            msg,
            "Permission denied (os error 13): /tmp/out/file.parquet"
        );
    }
}
