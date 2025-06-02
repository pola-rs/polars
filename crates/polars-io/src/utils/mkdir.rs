use std::io;
use std::path::Path;

pub fn mkdir_recursive(path: &Path) -> io::Result<()> {
    if path.starts_with("file:") {
        return Ok(());
    }

    std::fs::DirBuilder::new().recursive(true).create(
        path.parent()
            .ok_or(io::Error::other("path is not a file"))?,
    )
}

#[cfg(feature = "tokio")]
pub async fn tokio_mkdir_recursive(path: &Path) -> io::Result<()> {
    if path.starts_with("file:") {
        return Ok(());
    }

    tokio::fs::DirBuilder::new()
        .recursive(true)
        .create(
            path.parent()
                .ok_or(io::Error::other("path is not a file"))?,
        )
        .await
}
