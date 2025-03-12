use std::{fs, io};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyncOnCloseType {
    /// Don't call sync on close.
    #[default]
    None,

    /// Sync only the file contents.
    Data,
    /// Synce the file contents and the metadata.
    All,
}

pub fn sync_on_close(sync_on_close: SyncOnCloseType, file: &mut fs::File) -> io::Result<()> {
    match sync_on_close {
        SyncOnCloseType::None => Ok(()),
        SyncOnCloseType::Data => file.sync_data(),
        SyncOnCloseType::All => file.sync_all(),
    }
}

#[cfg(feature = "tokio")]
pub async fn tokio_sync_on_close(
    sync_on_close: SyncOnCloseType,
    file: &mut tokio::fs::File,
) -> io::Result<()> {
    match sync_on_close {
        SyncOnCloseType::None => Ok(()),
        SyncOnCloseType::Data => file.sync_data().await,
        SyncOnCloseType::All => file.sync_all().await,
    }
}
