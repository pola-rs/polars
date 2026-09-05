use std::io;
use std::path::{Path, PathBuf};

use polars_error::PolarsResult;
use polars_utils::io::_limit_path_len_io_err;
use polars_utils::pl_path::PlRefPath;

use crate::path_utils::resolve_homedir;

/// Creates the directory `path` will be written into.
///
/// Does nothing for paths with a scheme; object stores have no directories to create.
pub fn mkdir_recursive(path: &PlRefPath) -> PolarsResult<()> {
    let Some(parent) = writable_parent(path)? else {
        return Ok(());
    };

    std::fs::DirBuilder::new()
        .recursive(true)
        .create(&parent)
        .map_err(|err| _limit_path_len_io_err(&parent, err))
}

/// Async counterpart of [`mkdir_recursive`].
pub async fn tokio_mkdir_recursive(path: &PlRefPath) -> PolarsResult<()> {
    let Some(parent) = writable_parent(path)? else {
        return Ok(());
    };

    tokio::fs::DirBuilder::new()
        .recursive(true)
        .create(&parent)
        .await
        .map_err(|err| _limit_path_len_io_err(&parent, err))
}

/// The local directory that must exist before `path` can be opened for writing, or `None`
/// when there is nothing to create.
fn writable_parent(path: &PlRefPath) -> PolarsResult<Option<PathBuf>> {
    if path.has_scheme() {
        return Ok(None);
    }

    let Some(parent) = path.parent() else {
        return Err(io::Error::other(format!("path is not a file: {path}")).into());
    };

    // A path with a single component has no directory to create.
    if parent.is_empty() {
        return Ok(None);
    }

    Ok(Some(resolve_homedir(Path::new(parent)).into_owned()))
}
