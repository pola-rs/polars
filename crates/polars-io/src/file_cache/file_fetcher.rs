use polars_error::{PolarsError, PolarsResult};
use polars_utils::pl_path::PlRefPath;

use super::metadata::FileVersion;
use super::utils::last_modified_u64;
use crate::cloud::PolarsObjectStore;
use crate::pl_async;

pub trait FileFetcher: Send + Sync {
    fn get_uri(&self) -> &PlRefPath;
    fn fetch_metadata(&self) -> PolarsResult<RemoteMetadata>;
    /// Fetches the object to a `local_path`.
    fn fetch(&self, local_path: &std::path::Path) -> PolarsResult<()>;
    fn fetches_as_symlink(&self) -> bool;
}

pub struct RemoteMetadata {
    pub size: u64,
    pub(super) version: FileVersion,
}

/// A struct that fetches data from local disk and stores it into the `cache`.
/// Mostly used for debugging, it only ever gets called if `POLARS_FORCE_ASYNC` is set.
pub(super) struct LocalFileFetcher {
    path: PlRefPath,
}

impl LocalFileFetcher {
    pub(super) fn from_uri(uri: PlRefPath) -> Self {
        debug_assert_eq!(
            std::fs::canonicalize(uri.as_str())
                .ok()
                .and_then(|x| PlRefPath::try_from_pathbuf(x).ok())
                .as_ref(),
            Some(&uri),
        );

        Self { path: uri }
    }
}

impl FileFetcher for LocalFileFetcher {
    fn get_uri(&self) -> &PlRefPath {
        &self.path
    }

    fn fetches_as_symlink(&self) -> bool {
        #[cfg(target_family = "unix")]
        {
            true
        }
        #[cfg(not(target_family = "unix"))]
        {
            false
        }
    }

    fn fetch_metadata(&self) -> PolarsResult<RemoteMetadata> {
        let metadata = std::fs::metadata(&self.path).map_err(PolarsError::from)?;

        Ok(RemoteMetadata {
            size: metadata.len(),
            version: FileVersion::Timestamp(last_modified_u64(&metadata)),
        })
    }

    fn fetch(&self, local_path: &std::path::Path) -> PolarsResult<()> {
        #[cfg(target_family = "unix")]
        {
            std::os::unix::fs::symlink(&self.path, local_path).map_err(PolarsError::from)
        }
        #[cfg(not(target_family = "unix"))]
        {
            std::fs::copy(&self.path, local_path).map_err(PolarsError::from)?;
            Ok(())
        }
    }
}

pub(super) struct CloudFileFetcher {
    pub(super) uri: PlRefPath,
    pub(super) cloud_path: object_store::path::Path,
    pub(super) object_store: PolarsObjectStore,
}

impl FileFetcher for CloudFileFetcher {
    fn get_uri(&self) -> &PlRefPath {
        &self.uri
    }

    fn fetches_as_symlink(&self) -> bool {
        false
    }

    fn fetch_metadata(&self) -> PolarsResult<RemoteMetadata> {
        let metadata =
            pl_async::get_runtime().block_in_place_on(self.object_store.head(&self.cloud_path))?;

        Ok(RemoteMetadata {
            size: metadata.size as u64,
            version: metadata
                .e_tag
                .map(|x| FileVersion::ETag(blake3::hash(x.as_bytes()).to_hex()[..32].to_string()))
                .unwrap_or_else(|| {
                    FileVersion::Timestamp(metadata.last_modified.timestamp_millis() as u64)
                }),
        })
    }

    fn fetch(&self, local_path: &std::path::Path) -> PolarsResult<()> {
        pl_async::get_runtime().block_in_place_on(async {
            let file = &mut tokio::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(local_path)
                .await
                .map_err(PolarsError::from)?;

            self.object_store.download(&self.cloud_path, file).await
        })
    }
}
