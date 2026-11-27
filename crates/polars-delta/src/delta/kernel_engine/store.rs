use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{
    self, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use futures::stream::{self, BoxStream};
use polars_io_ext_ffi::common::{FfiBuf, FfiErrorKind};

use super::{HostError, HostIo, Live, MISSING_MODIFIED_MS, Source, closed, input};

pub(super) struct HostStore(pub(super) Arc<HostIo>);

impl fmt::Debug for HostStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Polars Delta host store")
    }
}

impl fmt::Display for HostStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Live {
    fn track<T>(&mut self, result: Result<T, HostError>) -> Result<T, HostError> {
        if let Err(error) = &result
            && error.kind == FfiErrorKind::OBJECT_CHANGED
            && self.changed.is_none()
        {
            self.changed = Some(error.clone());
        }
        result
    }

    pub(super) async fn pin(&mut self, url: &str, expected: Option<u64>) -> Result<u64, HostError> {
        let size = match self.pinned.get(url) {
            Some(&size) => size,
            None => {
                let result = self.storage.head(url).await.map_err(HostError::from);
                let size = self.track(result)?;
                self.pinned.insert(url.to_owned(), size);
                size
            },
        };
        if let Some(expected) = expected
            && expected != size
        {
            return self.track(Err(HostError {
                kind: FfiErrorKind::OBJECT_CHANGED,
                message: format!("delta: checkpoint size changed: expected {expected}, got {size}"),
            }));
        }
        Ok(size)
    }

    /// Fetch every object of `urls` at once. Failures are returned, not tracked: the reader that
    /// consumes an entry tracks it, so an unused failure never marks the table as changed.
    pub(super) async fn get_many(
        &self,
        urls: &[String],
    ) -> Vec<std::result::Result<FfiBuf, HostError>> {
        futures::future::join_all(
            urls.iter()
                .map(|url| async { self.storage.get(url).await.map_err(HostError::from) }),
        )
        .await
    }

    pub(super) async fn get(&mut self, url: &str) -> Result<FfiBuf, HostError> {
        let result = self.storage.get(url).await;
        self.track(result.map_err(HostError::from))
    }

    pub(super) async fn range(
        &mut self,
        url: &str,
        range: Range<u64>,
    ) -> Result<FfiBuf, HostError> {
        // StorageHandler ranges also arrive here. A cached pin takes no host I/O.
        self.pin(url, None).await?;
        let result = self.storage.range(url, range.start, range.end).await;
        self.track(result.map_err(HostError::from))
    }
}

#[async_trait::async_trait]
impl ObjectStore for HostStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let source = match location.as_ref().strip_prefix("_delta_log/") {
            Some(name) if !name.contains('/') && !name.ends_with(".crc") => Source {
                // Path::from_url_path already removed the synthetic URL's escaping.
                url: format!("{}{name}", self.0.prefix),
                expected_size: None,
                modified_ms: MISSING_MODIFIED_MS,
            },
            _ => self
                .0
                .aliases
                .lock()
                .unwrap()
                .get(location)
                .cloned()
                .ok_or_else(|| unsupported("get unregistered URL"))?,
        };
        if options.version.is_some()
            || options.if_match.is_some()
            || options.if_none_match.is_some()
            || options.if_modified_since.is_some()
            || options.if_unmodified_since.is_some()
        {
            return Err(unsupported("conditional get"));
        }

        let mut guard = self.0.live.lock().await;
        let live = guard.as_mut().ok_or_else(|| HostError::from(closed()))?;
        let url = &source.url;
        let full_get = !options.head && options.range.is_none();
        let (bytes, size, range) = if options.head {
            let size = live.pin(url, source.expected_size).await?;
            (Bytes::new(), size, 0..0)
        } else if let Some(requested) = options.range {
            let size = live.pin(url, source.expected_size).await?;
            let range = requested
                .as_range(size)
                .map_err(|error| object_store::Error::Generic {
                    store: "Polars Delta host",
                    source: Box::new(error),
                })?;
            let bytes = Bytes::from_owner(live.range(url, range.clone()).await?);
            (bytes, size, range)
        } else {
            let prefetched = self.0.prefetched.lock().unwrap().remove(url);
            let buf = match prefetched {
                Some(result) => live.track(result)?,
                None => live.get(url).await?,
            };
            let bytes = Bytes::from_owner(buf);
            let size = bytes.len() as u64;
            (bytes, size, 0..size)
        };
        drop(guard);

        let (payload, size, range) = if self.0.schema_only
            && full_get
            && let Some(name) = location.as_ref().strip_prefix("_delta_log/")
            && input::is_log_json(name)
        {
            let (payload, size) = input::schema_only_payload(name, bytes)?;
            (payload, size, 0..size)
        } else {
            (
                GetResultPayload::Stream(Box::pin(stream::iter([Ok(bytes)]))),
                size,
                range,
            )
        };

        Ok(GetResult {
            payload,
            meta: ObjectMeta {
                location: location.clone(),
                last_modified: chrono::DateTime::from_timestamp_millis(source.modified_ms)
                    .unwrap_or(chrono::DateTime::UNIX_EPOCH),
                size,
                e_tag: None,
                version: None,
            },
            range,
            attributes: Default::default(),
            extensions: Default::default(),
        })
    }

    async fn put_opts(
        &self,
        _: &Path,
        _: PutPayload,
        _: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(unsupported("put"))
    }

    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(unsupported("put multipart"))
    }

    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> object_store::Result<()> {
        Err(unsupported("copy"))
    }

    fn delete_stream(
        &self,
        _: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        Box::pin(stream::iter([Err(unsupported("delete"))]))
    }

    fn list(&self, _: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        Box::pin(stream::iter([Err(unsupported("list"))]))
    }

    async fn list_with_delimiter(&self, _: Option<&Path>) -> object_store::Result<ListResult> {
        Err(unsupported("list with delimiter"))
    }
}

fn unsupported(operation: &str) -> object_store::Error {
    object_store::Error::NotImplemented {
        operation: operation.to_owned(),
        implementer: "Polars Delta host store".to_owned(),
    }
}
