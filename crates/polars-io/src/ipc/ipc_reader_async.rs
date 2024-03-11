use std::ops::Range;
use std::sync::Arc;

use arrow::io::ipc::read::FileMetadata;
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use polars_core::frame::DataFrame;
use polars_error::{to_compute_err, PolarsResult};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard};

use crate::cloud::{build_object_store, CloudLocation, CloudOptions};
use crate::pl_async::{
    tune_with_concurrency_budget, with_concurrency_budget, MAX_BUDGET_PER_REQUEST,
};
use crate::prelude::IpcReader;

/// Polars specific wrapper for Arc<dyn ObjectStore> that limits the number of
/// concurrent requests for the entire application.
#[derive(Debug, Clone)]
pub struct PolarsObjectStore(Arc<dyn ObjectStore>);

impl<O> From<O> for PolarsObjectStore
where
    O: ObjectStore,
{
    fn from(value: O) -> Self {
        Self::new(Arc::new(value))
    }
}

impl PolarsObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self(store)
    }

    pub async fn get(&self, path: &Path) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || async {
            self
                .0
                .get(path)
                .await
                .map_err(to_compute_err)?
                .bytes()
                .await
                .map_err(to_compute_err)
        })
        .await
    }

    pub async fn get_range(&self, path: &Path, range: Range<usize>) -> PolarsResult<Bytes> {
        tune_with_concurrency_budget(1, || self.0.get_range(path, range))
            .await
            .map_err(to_compute_err)
    }

    pub async fn get_ranges(
        &self,
        path: &Path,
        ranges: &[Range<usize>],
    ) -> PolarsResult<Vec<Bytes>> {
        tune_with_concurrency_budget(
            (ranges.len() as u32).clamp(0, MAX_BUDGET_PER_REQUEST as u32),
            || self.0.get_ranges(path, ranges),
        )
        .await
        .map_err(to_compute_err)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(&self, path: &Path) -> PolarsResult<ObjectMeta> {
        with_concurrency_budget(1, || self.0.head(path))
            .await
            .map_err(to_compute_err)
    }
}

/// An Arrow IPC reader implemented on top of PolarsObjectStore.
pub struct IpcReaderAsync {
    store: PolarsObjectStore,
    path: Path,
    object_metadata: Mutex<Option<ObjectMeta>>,
    ipc_metadata: Mutex<Option<FileMetadata>>,
}

impl IpcReaderAsync {
    pub async fn from_uri(
        uri: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<IpcReaderAsync> {
        let (
            CloudLocation {
                prefix, expansion, ..
            },
            store,
        ) = build_object_store(uri, cloud_options).await?;

        Ok(Self {
            store: PolarsObjectStore::new(store),
            path: {
                // Any wildcards should already have been resolved here. Without this assertion they would
                // be ignored.
                debug_assert!(expansion.is_none(), "path should not contain wildcards");
                Path::from_url_path(prefix).map_err(to_compute_err)?
            },
            object_metadata: Default::default(),
            ipc_metadata: Default::default(),
        })
    }

    async fn fetch_object_metadata(&self) -> PolarsResult<ObjectMeta> {
        self.store.head(&self.path).await
    }

    async fn object_metadata(&self) -> PolarsResult<MappedMutexGuard<'_, ObjectMeta>> {
        // NOTE: We have to be careful of deadlocks because we wait here for a
        // `head` request to finish which waits for a semaphore. If that
        // semaphore is also used in places that call this function, we can get
        // a deadlock.
        let mut object_metadata = self.object_metadata.lock().await;
        if object_metadata.is_none() {
            *object_metadata = Some(self.fetch_object_metadata().await?);
        }
        Ok(MutexGuard::map(object_metadata, |x| {
            x.as_mut().unwrap_or_else(|| unreachable!())
        }))
    }

    async fn file_size(&self) -> PolarsResult<usize> {
        Ok(self.object_metadata().await?.size)
    }

    async fn fetch_ipc_metadata(&self) -> PolarsResult<FileMetadata> {
        let file_size = self.file_size().await?;

        // TODO: Do a larger request and hope that the entire footer is contained within it to save one round-trip.
        let footer_metadata =
            self.store
                .get_range(
                    &self.path,
                    file_size.checked_sub(FOOTER_METADATA_SIZE).ok_or_else(|| {
                        to_compute_err("ipc file size is smaller than the minimum")
                    })?..file_size,
                )
                .await?;

        let footer_size = deserialize_footer_metadata(
            footer_metadata
                .as_ref()
                .try_into()
                .map_err(to_compute_err)?,
        )?;

        let footer = self
            .store
            .get_range(
                &self.path,
                file_size
                    .checked_sub(FOOTER_METADATA_SIZE + footer_size)
                    .ok_or_else(|| {
                        to_compute_err("invalid ipc footer metadata: footer size too large")
                    })?..file_size,
            )
            .await?;

        arrow::io::ipc::read::deserialize_footer(
            footer.as_ref(),
            footer_size.try_into().map_err(to_compute_err)?,
        )
    }

    pub async fn metadata(&self) -> PolarsResult<MappedMutexGuard<'_, FileMetadata>> {
        let mut ipc_metadata = self.ipc_metadata.lock().await;
        if ipc_metadata.is_none() {
            *ipc_metadata = Some(self.fetch_ipc_metadata().await?)
        }
        Ok(MutexGuard::map(ipc_metadata, |x| {
            x.as_mut().unwrap_or_else(|| unreachable!())
        }))
    }

    pub async fn data(&self) -> PolarsResult<DataFrame> {
        let bytes = self.store.get(&self.path).await?;
        let reader =
            <IpcReader<_> as crate::SerReader<_>>::new(std::io::Cursor::new(bytes.as_ref()))
                .with_row_index(None) // TODO
                .with_n_rows(None) // TODO
                .with_columns(None) // TODO
                .with_projection(None); // TODO
        let predicate = None; // TODO
        let verbose = false; // TODO
        reader.finish_with_scan_ops(predicate, verbose)
    }
}

const FOOTER_METADATA_SIZE: usize = 10;

fn deserialize_footer_metadata(bytes: [u8; FOOTER_METADATA_SIZE]) -> PolarsResult<usize> {
    let footer_size: usize =
        i32::from_le_bytes(bytes[0..4].try_into().unwrap_or_else(|_| unreachable!()))
            .try_into()
            .map_err(to_compute_err)?;

    // TODO: Move to polars-arrow and deduplicate parsing of footer metadata in sync and async readers.
    if &bytes[4..] != b"ARROW1" {
        Err(to_compute_err("invalid ipc footer magic"))?
    }

    Ok(footer_size)
}
