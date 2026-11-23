//! Host functions of `delta.v1`: storage over `PolarsObjectStore`, executed on the Polars IO
//! runtime.
use std::collections::HashMap;
use std::ffi::c_void;
use std::future::Future;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{GetOptions, GetResult, GetResultPayload, ObjectStoreExt};
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::cloud::concurrency_config::ConcurrencyStrategy;
use polars_io::cloud::{
    CloudLocation, CloudOptions, PolarsObjectStore, build_object_store, object_path_from_str,
};
use polars_io::path_utils::POLARS_TEMP_DIR_BASE_PATH;
use polars_io::utils::byte_source::read_buffered;
use polars_io_ext_ffi::common::{
    FfiBuf, FfiError, FfiErrorKind, FfiFuture, FfiOption, FfiResult, FfiStr, FfiVec, FutureExt,
};
use polars_io_ext_ffi::delta_v1::{Host, ObjectMeta, StorageHandle};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use tokio::io::AsyncWriteExt;
use tokio::sync::OnceCell;

use crate::io::iceberg_plugin::metadata_cache::ScopedMetadataCache;

/// State shared with the plugin for the duration of one `plan` call; [`Host::ctx`] points to it.
pub(super) struct HostCtx {
    /// `storage_options` and credential provider of the scan.
    pub cloud_options: Option<CloudOptions>,
    pub metadata_cache: Option<ScopedMetadataCache>,
    pub io_error_kinds: IoErrorKinds,
}

impl HostCtx {
    /// The host functions. `self` must outlive the plugin call.
    pub fn host(&self) -> Host {
        Host {
            ctx: self as *const Self as *const c_void,
            get_storage: host_get_storage,
            storage_get,
            storage_head,
            storage_list,
            storage_get_range,
            storage_release,
            log: host_log,
        }
    }
}

/// Kinds of the IO errors of the scan's storage calls, by message. The contract only
/// distinguishes `NOT_FOUND` and `IO`; this restores e.g. `PermissionDenied` when the plugin
/// fails with one of them.
#[derive(Clone, Default)]
pub(super) struct IoErrorKinds(Arc<Mutex<Vec<(String, std::io::ErrorKind)>>>);

impl IoErrorKinds {
    /// The kind of the storage error a plugin error message was made from: the plugin adds
    /// context around the message.
    pub fn kind_of(&self, plugin_msg: &str) -> Option<std::io::ErrorKind> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(msg, _)| plugin_msg.contains(msg.as_str()))
            .max_by_key(|(msg, _)| msg.len())
            .map(|(_, kind)| *kind)
    }

    fn to_ffi_err(&self, e: PolarsError) -> FfiError {
        let e = polars_to_ffi_err_kind(e);
        if let Some(kind) = e.1 {
            self.0
                .lock()
                .unwrap()
                .push((e.0.message().to_string(), kind));
        }
        e.0
    }
}

fn polars_to_ffi_err(e: PolarsError) -> FfiError {
    polars_to_ffi_err_kind(e).0
}

/// Also returns the kind of an IO error.
fn polars_to_ffi_err_kind(e: PolarsError) -> (FfiError, Option<std::io::ErrorKind>) {
    let io_kind = match &e {
        PolarsError::IO { error, .. } => Some(error.kind()),
        _ => None,
    };
    let kind = match io_kind {
        Some(std::io::ErrorKind::NotFound) => FfiErrorKind::NOT_FOUND,
        Some(_) => FfiErrorKind::IO,
        None => FfiErrorKind::OTHER,
    };
    (FfiError::new(kind, e.to_string()), io_kind)
}

fn str_arg(s: FfiStr) -> Result<String, FfiError> {
    s.as_str().map(str::to_owned).map_err(|e| {
        FfiError::new(
            FfiErrorKind::INVALID_ARGUMENT,
            format!("argument is not UTF-8: {e}"),
        )
    })
}

/// Run `fut` on the Polars IO runtime. Dropping the returned future aborts the task.
fn spawn_io<T: Send + 'static>(
    fut: impl Future<Output = Result<T, FfiError>> + Send + 'static,
) -> FfiFuture<FfiResult<T>> {
    let handle = AbortOnDropHandle(ASYNC.spawn(fut));

    async move {
        match handle.await {
            Ok(result) => result.into(),
            Err(join_err) => FfiResult::Err(FfiError::new(
                FfiErrorKind::OTHER,
                format!("host IO task failed: {join_err}"),
            )),
        }
    }
    .into_ffi()
}

fn ready<T: Send + 'static>(result: Result<T, FfiError>) -> FfiFuture<FfiResult<T>> {
    async move { FfiResult::from(result) }.into_ffi()
}

// `extern "C"` entry points called by the plugin. Unwinding across them would abort the process,
// so panics are turned into errors. Panics inside the returned futures are caught by the IO task
// (`spawn_io`).
fn guard_future<T: Send + 'static>(
    name: &'static str,
    f: impl FnOnce() -> FfiFuture<FfiResult<T>>,
) -> FfiFuture<FfiResult<T>> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        ready(Err(FfiError::new(
            FfiErrorKind::PANIC,
            format!("Polars host function '{name}' panicked"),
        )))
    })
}

unsafe fn host_ctx<'a>(ctx: *const c_void) -> &'a HostCtx {
    // SAFETY: `ctx` is the `HostCtx` of the current plugin call, which the caller keeps alive
    // until the call returns.
    unsafe { &*(ctx as *const HostCtx) }
}

unsafe extern "C" fn host_get_storage(
    ctx: *const c_void,
    url: FfiStr,
) -> FfiFuture<FfiResult<StorageHandle>> {
    guard_future("get_storage", || {
        let ctx = unsafe { host_ctx(ctx) };
        ready((|| {
            // Validated here so that errors surface at `get_storage` rather than at first use.
            let url = str_arg(url)?;
            PlRefPath::new(&url)
                .to_absolute_path()
                .map_err(polars_to_ffi_err)?;

            static NEXT_RESOLUTION: AtomicU64 = AtomicU64::new(0);
            let storage = Arc::new(HostStorage {
                cloud_options: ctx.cloud_options.clone(),
                metadata_cache: ctx.metadata_cache.clone(),
                io_error_kinds: ctx.io_error_kinds.clone(),
                resolution: NEXT_RESOLUTION.fetch_add(1, Ordering::Relaxed),
                listed: Default::default(),
                checkpoints: Default::default(),
            });
            Ok(StorageHandle(Arc::into_raw(storage) as *const c_void))
        })())
    })
}

unsafe extern "C" fn host_log(_ctx: *const c_void, msg: FfiStr) {
    // Logging must never unwind into the plugin.
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        eprintln!("{}", msg.as_str().unwrap_or("<non-UTF-8 message>"));
    }));
}

struct HostStorage {
    cloud_options: Option<CloudOptions>,
    metadata_cache: Option<ScopedMetadataCache>,
    io_error_kinds: IoErrorKinds,
    resolution: u64,
    /// Strong cloud validators from LIST, for whole-object reads of commit and checkpoint JSON.
    listed: Mutex<HashMap<String, Generation>>,
    /// The generation to keep for each HEAD followed by range reads, including multipart passes.
    checkpoints: Mutex<HashMap<String, Arc<OnceCell<CheckpointRead>>>>,
}

struct ResolvedLocation {
    store: PolarsObjectStore,
    path: object_store::path::Path,
    is_local: bool,
}

#[derive(Clone, Debug)]
enum Generation {
    ETag(String),
    Version(String),
}

impl Generation {
    fn from_meta(meta: &object_store::ObjectMeta) -> Option<Self> {
        // Prefer a strong ETag so that S3 readers with GetObject but no GetObjectVersion can
        // still read checkpoints on versioned buckets. S3's null version can be overwritten.
        if let Some(etag) = &meta.e_tag
            && !etag.is_empty()
            && etag != "*"
            && !etag.starts_with("W/")
        {
            Some(Self::ETag(etag.clone()))
        } else {
            meta.version
                .as_ref()
                .filter(|v| !v.is_empty() && v.as_str() != "null")
                .map(|v| Self::Version(v.clone()))
        }
    }

    fn matches(&self, meta: &object_store::ObjectMeta) -> bool {
        match self {
            Self::ETag(tag) => meta.e_tag.as_ref() == Some(tag),
            Self::Version(version) => meta.version.as_ref() == Some(version),
        }
    }
}

struct CheckpointRead {
    size: u64,
    source: CheckpointSource,
}

enum CheckpointSource {
    // An open local file or an anonymous temporary copy of one cloud response.
    File(Arc<std::fs::File>),
    Ranges(ResolvedLocation, Option<Generation>),
}

impl HostStorage {
    fn cache_key(
        &self,
        scope: &str,
        url: &str,
        generation: Option<&Generation>,
        range: Option<&Range<u64>>,
    ) -> String {
        match generation {
            Some(generation) => format!("delta:{scope:?}:{url:?}:{generation:?}:{range:?}"),
            None => format!("delta:{scope:?}:{url:?}:{}:{range:?}", self.resolution),
        }
    }

    async fn resolve(&self, url: &str) -> PolarsResult<ResolvedLocation> {
        let url = PlRefPath::new(url);
        let is_local = matches!(
            url.scheme(),
            None | Some(CloudScheme::File | CloudScheme::FileNoHostname)
        );
        let (CloudLocation { prefix, .. }, store) =
            build_object_store(url, self.cloud_options.as_ref(), false).await?;

        // Object store paths have no `.`, `..` or empty segments, which local paths (e.g.
        // `/wh/./t/metadata/...` from a table location) can have. Object keys are kept as they
        // are: `a//b` is another key than `a/b`.
        let path = if is_local {
            object_path_from_str(&resolve_local_path(&prefix)?)?
        } else {
            object_path_from_str(&prefix)?
        };

        Ok(ResolvedLocation {
            store,
            path,
            is_local,
        })
    }

    async fn open_checkpoint(&self, url: &str) -> PolarsResult<CheckpointRead> {
        let loc = self.resolve(url).await?;
        let classic = url
            .rsplit('/')
            .next()
            .and_then(|name| name.strip_suffix(".checkpoint.parquet"))
            .is_some_and(|version| {
                version.len() == 20 && version.bytes().all(|b| b.is_ascii_digit())
            });
        if classic && loc.is_local {
            return open_checkpoint_file(&loc).await;
        }

        let meta = loc
            .store
            .head(&loc.path, ConcurrencyStrategy::BytesBased)
            .await?;
        let generation = Generation::from_meta(&meta);
        if classic && generation.is_none() {
            return open_checkpoint_file(&loc).await;
        }
        Ok(CheckpointRead {
            size: meta.size,
            source: CheckpointSource::Ranges(loc, generation),
        })
    }
}

async fn open_checkpoint_file(loc: &ResolvedLocation) -> PolarsResult<CheckpointRead> {
    let path = &loc.path;
    loc.store
        .exec_with_rebuild_retry_on_err(|s| async move {
            let result = s.get(path).await?;
            let size = result.meta.size;
            let file = match result.payload {
                GetResultPayload::File(file, _) => file,
                _ => stage_checkpoint(result).await?,
            };
            Ok(CheckpointRead {
                size,
                source: CheckpointSource::File(Arc::new(file)),
            })
        })
        .await
}

async fn stage_checkpoint(result: GetResult) -> object_store::Result<std::fs::File> {
    fn io_error(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> object_store::Error {
        object_store::Error::Generic {
            store: "staging Delta checkpoint",
            source: error.into(),
        }
    }

    // A response without a validator can still provide one generation. Keep its body on disk
    // so a projected or schema-only read never needs to buffer an entire checkpoint.
    let size = result.meta.size;
    let file = ASYNC
        .spawn_blocking(|| tempfile::tempfile_in(&*POLARS_TEMP_DIR_BASE_PATH))
        .await
        .map_err(io_error)?
        .map_err(io_error)?;
    let mut file = tokio::fs::File::from_std(file);
    let mut stream = result.into_stream();
    let mut written = 0;
    while let Some(chunk) = stream.try_next().await? {
        file.write_all(&chunk).await.map_err(io_error)?;
        written += chunk.len() as u64;
    }
    if written != size {
        return Err(io_error(
            "checkpoint response length did not match its size",
        ));
    }
    file.flush().await.map_err(io_error)?;
    Ok(file.into_std().await)
}

async fn fetch_bytes(
    loc: &ResolvedLocation,
    generation: Option<&Generation>,
    range: Option<Range<u64>>,
) -> PolarsResult<Bytes> {
    let path = &loc.path;
    loc.store
        .exec_with_rebuild_retry_on_err(|s| {
            let mut options = GetOptions {
                range: range.clone().map(Into::into),
                ..Default::default()
            };
            match generation {
                Some(Generation::ETag(tag)) => options.if_match = Some(tag.clone()),
                Some(Generation::Version(version)) => options.version = Some(version.clone()),
                None => {},
            }
            async move {
                let result = s.get_opts(path, options).await?;
                // Some HTTP servers accept conditional requests without enforcing them.
                if generation.is_some_and(|generation| !generation.matches(&result.meta)) {
                    return Err(object_store::Error::Precondition {
                        path: path.to_string(),
                        source: "object generation does not match".into(),
                    });
                }
                result.bytes().await
            }
        })
        .await
}

fn read_error(io_error_kinds: &IoErrorKinds, error: PolarsError, pinned: bool) -> FfiError {
    if pinned
        && std::iter::successors(Some(&error as &dyn std::error::Error), |e| e.source()).any(|e| {
            // io::Error can expose the object-store error directly as its source.
            matches!(
                e.downcast_ref::<object_store::Error>(),
                Some(
                    object_store::Error::Precondition { .. } | object_store::Error::NotFound { .. }
                )
            )
        })
    {
        return FfiError::new(FfiErrorKind::OBJECT_CHANGED, error.to_string());
    }
    io_error_kinds.to_ffi_err(error)
}

/// `path` with its `..` segments applied as the filesystem applies them. On Unix, `link/..` is
/// the parent of the symlink's target rather than the directory holding `link`, so a path with
/// `..` is resolved by the filesystem. A path that does not resolve (e.g. a missing file) is an
/// error: its lexical normalization can name another, existing file.
fn resolve_local_path(path: &str) -> PolarsResult<String> {
    #[cfg(unix)]
    if path.split('/').any(|s| s == "..") {
        let resolved = std::fs::canonicalize(path).map_err(|e| PolarsError::IO {
            msg: Some(format!("{e}: {path}").into()),
            error: Arc::new(e),
        })?;
        return resolved
            .into_os_string()
            .into_string()
            .map_err(|p| polars_err!(ComputeError: "resolved path is not UTF-8: {}", p.display()));
    }

    Ok(normalize_local_path(path))
}

/// `path` without `.` and empty segments, and with `..` segments applied lexically, as Windows
/// applies them.
fn normalize_local_path(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();

    for segment in path.split('/') {
        match segment {
            "" | "." => {},
            ".." => {
                segments.pop();
            },
            segment => segments.push(segment),
        }
    }

    let mut out = segments.join("/");

    if path.starts_with('/') {
        out.insert(0, '/');
    }

    out
}

unsafe fn storage_arc(storage: StorageHandle) -> Arc<HostStorage> {
    // SAFETY: The handle comes from `Arc::into_raw` in `host_get_storage` and has not been
    // released.
    unsafe {
        Arc::increment_strong_count(storage.0 as *const HostStorage);
        Arc::from_raw(storage.0 as *const HostStorage)
    }
}

unsafe extern "C" fn storage_release(storage: StorageHandle) {
    unsafe { Arc::decrement_strong_count(storage.0 as *const HostStorage) };
}

fn is_cacheable(url: &str) -> bool {
    // Mutable hints and listings always go back to storage.
    let name = url.rsplit('/').next().unwrap_or(url);
    url.contains("/_delta_log/")
        && name != "_last_checkpoint"
        && (name.ends_with(".json") || name.ends_with(".parquet"))
}

unsafe extern "C" fn storage_get(
    storage: StorageHandle,
    url: FfiStr,
) -> FfiFuture<FfiResult<FfiBuf>> {
    guard_future("storage_get", || {
        let storage = unsafe { storage_arc(storage) };
        let url = str_arg(url);

        spawn_io(async move {
            let url = url?;
            let generation = storage.listed.lock().unwrap().get(&url).cloned();
            let fetch = || async {
                let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
                let loc = storage.resolve(&url).await.map_err(to_ffi_err)?;
                fetch_bytes(&loc, generation.as_ref(), None)
                    .await
                    .map_err(|e| read_error(&storage.io_error_kinds, e, generation.is_some()))
            };

            let bytes = match &storage.metadata_cache {
                Some(c) if is_cacheable(&url) => {
                    let key = storage.cache_key(&c.scope, &url, generation.as_ref(), None);
                    c.cache.get_or_fetch(&key, &c.stats, fetch).await?
                },
                _ => fetch().await?,
            };

            Ok(FfiBuf::from_owner(bytes))
        })
    })
}

unsafe extern "C" fn storage_head(
    storage: StorageHandle,
    url: FfiStr,
) -> FfiFuture<FfiResult<u64>> {
    guard_future("storage_head", || {
        let storage = unsafe { storage_arc(storage) };
        let url = str_arg(url);

        spawn_io(async move {
            let url = url?;
            let checkpoint = storage
                .checkpoints
                .lock()
                .unwrap()
                .entry(url.clone())
                .or_default()
                .clone();
            let read = checkpoint
                .get_or_try_init(|| storage.open_checkpoint(&url))
                .await
                .map_err(|e| storage.io_error_kinds.to_ffi_err(e))?;
            Ok(read.size)
        })
    })
}

unsafe extern "C" fn storage_get_range(
    storage: StorageHandle,
    url: FfiStr,
    start: u64,
    end: u64,
) -> FfiFuture<FfiResult<FfiBuf>> {
    guard_future("storage_get_range", || {
        let storage = unsafe { storage_arc(storage) };
        let url = str_arg(url);
        spawn_io(async move {
            let len = end
                .checked_sub(start)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    FfiError::new(FfiErrorKind::INVALID_ARGUMENT, "invalid byte range")
                })?;
            if len == 0 {
                return Ok(FfiBuf::from_vec(Vec::new()));
            }
            let url = url?;
            let checkpoint = storage.checkpoints.lock().unwrap().get(&url).cloned();
            let read = checkpoint.as_deref().and_then(OnceCell::get);
            if read.is_some_and(|read| end > read.size) {
                return Err(FfiError::new(
                    FfiErrorKind::IO,
                    "range exceeds checkpoint size",
                ));
            }
            let source = read.map(|read| &read.source);
            let generation = match source {
                Some(CheckpointSource::Ranges(loc, generation)) if !loc.is_local => {
                    generation.as_ref()
                },
                _ => None,
            };
            let range = start..end;
            let fetch = || async {
                match source {
                    Some(CheckpointSource::File(file)) => {
                        let file = file.clone();
                        ASYNC
                            .spawn_blocking(move || {
                                read_buffered(&file, start, len).map(Bytes::from_owner)
                            })
                            .await
                            .map_err(|e| FfiError::new(FfiErrorKind::OTHER, e.to_string()))?
                            .map_err(|e| storage.io_error_kinds.to_ffi_err(e))
                    },
                    Some(CheckpointSource::Ranges(loc, generation)) => {
                        fetch_bytes(loc, generation.as_ref(), Some(range.clone()))
                            .await
                            .map_err(|e| {
                                read_error(&storage.io_error_kinds, e, generation.is_some())
                            })
                    },
                    None => {
                        let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
                        let loc = storage.resolve(&url).await.map_err(to_ffi_err)?;
                        fetch_bytes(&loc, None, Some(range.clone()))
                            .await
                            .map_err(to_ffi_err)
                    },
                }
            };
            let bytes = match &storage.metadata_cache {
                Some(c) if is_cacheable(&url) => {
                    let key = storage.cache_key(&c.scope, &url, generation, Some(&range));
                    c.cache.get_or_fetch(&key, &c.stats, fetch).await?
                },
                _ => fetch().await?,
            };
            if bytes.len() != len {
                return Err(FfiError::new(FfiErrorKind::IO, "short storage range read"));
            }
            Ok(FfiBuf::from_owner(bytes))
        })
    })
}

unsafe extern "C" fn storage_list(
    storage: StorageHandle,
    prefix: FfiStr,
    start_after: FfiOption<FfiStr>,
) -> FfiFuture<FfiResult<FfiVec<ObjectMeta>>> {
    guard_future("storage_list", || {
        let storage = unsafe { storage_arc(storage) };
        let prefix = str_arg(prefix);
        let start_after = start_after.into_option().map(str_arg).transpose();
        spawn_io(async move {
            let prefix = prefix?;
            let start_after = start_after?;
            let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
            let loc = storage.resolve(&prefix).await.map_err(to_ffi_err)?;
            let path = &loc.path;
            // Every key of the table's log from `start_after` on, which a store lists from an
            // offset in one request. A local directory walk is no cheaper with one. Offset listings
            // are recursive, so entries below `_sidecars/` and `_commits/` are dropped.
            let listing = match &start_after {
                Some(start) if !loc.is_local => {
                    let offset = prefix_offset(&loc.path, &prefix, start);
                    loc.store
                        .exec_with_rebuild_retry_on_err(|s| {
                            let offset = offset.clone();
                            async move {
                                s.list_with_offset(Some(path), &offset)
                                    .try_filter(|m| {
                                        std::future::ready(
                                            m.location
                                                .prefix_match(path)
                                                .is_some_and(|mut parts| parts.nth(1).is_none()),
                                        )
                                    })
                                    .try_collect::<Vec<_>>()
                                    .await
                            }
                        })
                        .await
                        .map_err(to_ffi_err)?
                },
                _ => {
                    loc.store
                        .exec_with_rebuild_retry_on_err(|s| async move {
                            s.list_with_delimiter(Some(path)).await
                        })
                        .await
                        .map_err(to_ffi_err)?
                        .objects
                },
            };
            let mut objects = Vec::with_capacity(listing.len());
            let mut identities = Vec::with_capacity(listing.len());
            for object in listing {
                // Prefix lists only direct children; keep the caller's URL authority/path form.
                let Some(name) = object.location.as_ref().rsplit('/').next() else {
                    continue;
                };
                let url = format!("{}/{name}", prefix.trim_end_matches('/'));
                if start_after.as_ref().is_some_and(|start| url <= *start) {
                    continue;
                }
                let modified = object.last_modified.timestamp_millis();
                if !loc.is_local
                    && is_cacheable(&url)
                    && let Some(generation) = Generation::from_meta(&object)
                {
                    identities.push((url.clone(), generation));
                }
                objects.push(ObjectMeta {
                    url: FfiBuf::from_string(url),
                    size: object.size,
                    modified_ms: Some(modified).into(),
                });
            }
            storage.listed.lock().unwrap().extend(identities);
            objects.sort_unstable_by(|a, b| a.url.as_slice().cmp(b.url.as_slice()));
            Ok(FfiVec::from_vec(objects))
        })
    })
}

/// The object-store key `start` (a URL in the caller's form, under `prefix`) names, for an
/// exclusive offset listing of `dir`.
fn prefix_offset(
    dir: &object_store::path::Path,
    prefix: &str,
    start: &str,
) -> object_store::path::Path {
    let name = start
        .strip_prefix(prefix.trim_end_matches('/'))
        .unwrap_or(start)
        .trim_start_matches('/');
    dir.clone().join(name)
}

#[cfg(test)]
mod tests {
    use polars_io::cloud::ObjectStoreErrorContext;

    use super::*;

    #[test]
    fn pinned_read_converts_wrapped_precondition_to_object_changed() {
        let path = "s3://bucket/checkpoint.parquet";
        let error = ObjectStoreErrorContext::new(PlRefPath::new(path)).attach_err_info(
            object_store::Error::Precondition {
                path: path.to_owned(),
                source: "etag mismatch".into(),
            },
        );
        let error = PolarsError::from(error);
        let kinds = IoErrorKinds::default();
        assert_eq!(
            read_error(&kinds, error.clone(), true).kind(),
            FfiErrorKind::OBJECT_CHANGED
        );
        assert_eq!(read_error(&kinds, error, false).kind(), FfiErrorKind::IO);
    }

    #[test]
    fn checkpoint_stream_can_be_read_positionally_after_staging() {
        let result = GetResult {
            payload: GetResultPayload::Stream(Box::pin(futures::stream::iter([
                Ok(Bytes::from_static(b"abcd")),
                Ok(Bytes::from_static(b"efgh")),
            ]))),
            meta: object_store::ObjectMeta {
                location: "checkpoint.parquet".into(),
                last_modified: chrono::DateTime::UNIX_EPOCH,
                size: 8,
                e_tag: None,
                version: None,
            },
            range: 0..8,
            attributes: Default::default(),
            extensions: Default::default(),
        };
        let file = ASYNC.block_on(stage_checkpoint(result)).unwrap();
        assert_eq!(read_buffered(&file, 2, 4).unwrap().as_ref(), b"cdef");
    }
}
