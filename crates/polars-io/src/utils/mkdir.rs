use std::io;
use std::path::Path;
use polars_utils::address::AddressRef;

pub fn mkdir_recursive(addr: AddressRef<'_>) -> io::Result<()> {
    if let Some(path) = addr.as_local_path() {
        std::fs::DirBuilder::new().recursive(true).create(
            path.parent()
                .ok_or(io::Error::other("path is not a file"))?,
        )
    } else {
        Ok(())
    }
}

#[cfg(feature = "tokio")]
pub async fn tokio_mkdir_recursive(addr: AddressRef<'_>) -> io::Result<()> {
    if let Some(path) = addr.as_local_path() {
        tokio::fs::DirBuilder::new()
            .recursive(true)
            .create(
                path.parent()
                    .ok_or(io::Error::other("path is not a file"))?,
            )
            .await
    } else {
        Ok(())
    }
}
