use std::io;

use polars_utils::pl_path::PlRefPath;

pub fn mkdir_recursive(path: &PlRefPath) -> io::Result<()> {
    if !path.has_scheme() {
        std::fs::DirBuilder::new().recursive(true).create(
            path.parent()
                .ok_or(io::Error::other("path is not a file"))?,
        )?;
    }

    Ok(())
}

#[cfg(feature = "tokio")]
pub async fn tokio_mkdir_recursive(path: &PlRefPath) -> io::Result<()> {
    if !path.has_scheme() {
        tokio::fs::DirBuilder::new()
            .recursive(true)
            .create(
                path.parent()
                    .ok_or(io::Error::other("path is not a file"))?,
            )
            .await?;
    }

    Ok(())
}
