use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum FileVersion {
    Timestamp(u64),
    ETag(String),
    Uninitialized,
}

#[derive(Debug)]
pub enum LocalCompareError {
    LastModifiedMismatch { expected: u64, actual: u64 },
    SizeMismatch { expected: u64, actual: u64 },
    DataFileReadError(std::io::Error),
}

pub type LocalCompareResult = Result<(), LocalCompareError>;

/// Metadata written to a file used to track state / synchronize across processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct EntryMetadata {
    pub(super) uri: Arc<str>,
    pub(super) local_last_modified: u64,
    pub(super) local_size: u64,
    pub(super) remote_version: FileVersion,
    /// TTL since last access, in seconds.
    pub(super) ttl: u64,
}

impl std::fmt::Display for LocalCompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LastModifiedMismatch { expected, actual } => write!(
                f,
                "last modified time mismatch: expected {}, found {}",
                expected, actual
            ),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {}, found {}", expected, actual)
            },
            Self::DataFileReadError(err) => {
                write!(f, "failed to read local file metadata: {}", err)
            },
        }
    }
}

impl EntryMetadata {
    pub(super) fn new(uri: Arc<str>, ttl: u64) -> Self {
        Self {
            uri,
            local_last_modified: 0,
            local_size: 0,
            remote_version: FileVersion::Uninitialized,
            ttl,
        }
    }

    pub(super) fn compare_local_state(&self, data_file_path: &Path) -> LocalCompareResult {
        let metadata = match std::fs::metadata(data_file_path) {
            Ok(v) => v,
            Err(e) => return Err(LocalCompareError::DataFileReadError(e)),
        };

        let local_last_modified = super::utils::last_modified_u64(&metadata);
        let local_size = metadata.len();

        if local_last_modified != self.local_last_modified {
            Err(LocalCompareError::LastModifiedMismatch {
                expected: self.local_last_modified,
                actual: local_last_modified,
            })
        } else if local_size != self.local_size {
            Err(LocalCompareError::SizeMismatch {
                expected: self.local_size,
                actual: local_size,
            })
        } else {
            Ok(())
        }
    }

    pub(super) fn try_write<W: std::io::Write>(&self, writer: &mut W) -> serde_json::Result<()> {
        serde_json::to_writer(writer, self)
    }

    pub(super) fn try_from_reader<R: std::io::Read>(reader: &mut R) -> serde_json::Result<Self> {
        serde_json::from_reader(reader)
    }
}
