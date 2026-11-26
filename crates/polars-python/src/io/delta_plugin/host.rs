//! Host functions of `delta.v1`: storage over `PolarsObjectStore`, executed on the Polars IO
//! runtime.
use std::collections::HashMap;
use std::ffi::c_void;
use std::future::Future;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use tokio::sync::{OnceCell, OwnedMutexGuard, Semaphore};

use crate::io::iceberg_plugin::metadata_cache::ScopedMetadataCache;

const PINNED_LOCAL_CHECKPOINTS: usize = 32;

/// State shared with the plugin for the duration of one `plan` call; [`Host::ctx`] points to it.
pub(super) struct HostCtx {
    /// `storage_options` and credential provider of the scan.
    pub cloud_options: Option<CloudOptions>,
    pub metadata_cache: Option<ScopedMetadataCache>,
    pub io_error_kinds: IoErrorKinds,
    /// The version the plan is pinned to; `_last_checkpoint` may be served from the cache only
    /// then. Unpinned (latest) plans always read it from storage.
    pub pinned_version: Option<i64>,
    /// Whether a plan served a cached `_last_checkpoint`; cleared by a retry that bypasses it.
    pub hint_served: Arc<AtomicBool>,
    /// The `_last_checkpoint` this plan read from storage, as (cache key, bytes; empty for a 404).
    /// Recorded by [`record_hint`] once the plan has resolved its version.
    pub hint_read: Arc<Mutex<Option<(String, Bytes)>>>,
}

/// Remember the `_last_checkpoint` a plan read, for pinned plans of the same `version`. A pinned
/// plan of another version reads it fresh: a scan over a table that advances would otherwise
/// keep planning from an ever older checkpoint.
pub fn record_hint(cache: &ScopedMetadataCache, read: (String, Bytes), version: i64) {
    let (key, bytes) = read;
    let bytes = if bytes.is_empty() {
        bytes
    } else {
        match strip_checkpoint_schemas(&bytes) {
            // Kernel would ignore a hint newer than the version for a backward scan.
            Some((hint, hint_version)) if hint_version <= version => hint,
            _ => return,
        }
    };
    let mut entry = version.to_be_bytes().to_vec();
    entry.extend_from_slice(&bytes);
    cache.cache.replace(&key, Bytes::from(entry));
}

/// Make the entry of `read` match no pinned plan: a plan and its retry both failed.
pub fn poison_hint(cache: &ScopedMetadataCache, read: (String, Bytes)) {
    cache.cache.replace(&read.0, Bytes::new());
}

/// The hint without `checkpointSchema` and `sidecarFileSchema`, and its version. Kernel trusts
/// them when the version and shape match, and a checkpoint rewritten at that version by another
/// writer can differ: Kernel reads the footer instead, which the cache keys by generation.
fn strip_checkpoint_schemas(bytes: &[u8]) -> Option<(Bytes, i64)> {
    let mut hint: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let version = hint.get("version")?.as_i64()?;
    let map = hint.as_object_mut()?;
    map.remove("checkpointSchema");
    map.remove("sidecarFileSchema");
    Some((Bytes::from(serde_json::to_vec(&hint).ok()?), version))
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
                pinned_version: ctx.pinned_version,
                hint_served: ctx.hint_served.clone(),
                hint_read: ctx.hint_read.clone(),
                resolution: NEXT_RESOLUTION.fetch_add(1, Ordering::Relaxed),
                listed: Default::default(),
                checkpoints: Default::default(),
                local_file_slots: Semaphore::new(PINNED_LOCAL_CHECKPOINTS),
                stage_lock: Default::default(),
                checkpoint_spool: Default::default(),
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
    pinned_version: Option<i64>,
    hint_served: Arc<AtomicBool>,
    hint_read: Arc<Mutex<Option<(String, Bytes)>>>,
    resolution: u64,
    /// Strong cloud validators and sizes from LIST, for whole-object reads and checkpoint opens.
    listed: Mutex<HashMap<String, (Generation, u64)>>,
    /// The generation to keep for each HEAD followed by range reads, including multipart passes.
    checkpoints: Mutex<HashMap<String, Arc<OnceCell<CheckpointRead>>>>,
    local_file_slots: Semaphore,
    stage_lock: Arc<tokio::sync::Mutex<()>>,
    checkpoint_spool: OnceCell<Arc<CheckpointSpool>>,
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
        if let Some(tag) = meta.e_tag.as_deref().and_then(strong_etag) {
            Some(Self::ETag(tag.to_owned()))
        } else {
            meta.version
                .as_ref()
                .filter(|v| !v.is_empty() && v.as_str() != "null")
                .map(|v| Self::Version(v.clone()))
        }
    }

    fn matches(&self, meta: &object_store::ObjectMeta) -> bool {
        match self {
            Self::ETag(tag) => meta.e_tag.as_deref().and_then(strong_etag) == Some(tag.as_str()),
            Self::Version(version) => meta.version.as_ref() == Some(version),
        }
    }
}

/// Providers can leave the quotes off the opaque tag in LIST metadata.
fn strong_etag(tag: &str) -> Option<&str> {
    let tag = if let Some(tag) = tag.strip_prefix('"') {
        tag.strip_suffix('"')?
    } else if tag.is_empty()
        || tag == "*"
        || tag.starts_with("W/")
        || tag.starts_with("w/")
        || tag.contains(',')
    {
        return None;
    } else {
        tag
    };
    tag.bytes()
        .all(|b| matches!(b, b'!' | b'#'..=b'~'))
        .then_some(tag)
}

struct CheckpointRead {
    size: u64,
    source: CheckpointSource,
}

enum CheckpointSource {
    File(CheckpointFile),
    Ranges(ResolvedLocation, Generation),
}

#[derive(Clone)]
enum CheckpointFile {
    Local(Arc<std::fs::File>),
    Staged(Arc<CheckpointSpool>, u64),
}

impl CheckpointFile {
    fn read_at(&self, offset: u64, len: usize) -> PolarsResult<Bytes> {
        match self {
            Self::Local(file) => read_buffered(file, offset, len).map(Bytes::from_owner),
            Self::Staged(spool, base) => spool.read_at(base + offset, len),
        }
    }
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

    async fn open_checkpoint(&self, url: &str) -> Result<CheckpointRead, FfiError> {
        let loc = self
            .resolve(url)
            .await
            .map_err(|e| self.io_error_kinds.to_ffi_err(e))?;
        if loc.is_local {
            return self
                .open_checkpoint_file(&loc, None)
                .await
                .map_err(|e| self.io_error_kinds.to_ffi_err(e));
        }

        let listed = self.listed.lock().unwrap().get(url).cloned();
        if let Some((generation, size)) = listed {
            // LIST gave the generation and size: no HEAD. Every range read is conditional on the
            // generation (and re-checks the response's), so a replaced object fails as
            // OBJECT_CHANGED, as a mismatch between LIST and HEAD did.
            return Ok(CheckpointRead {
                size,
                source: CheckpointSource::Ranges(loc, generation),
            });
        }
        let meta = loc
            .store
            .head(&loc.path, ConcurrencyStrategy::BytesBased)
            .await
            .map_err(|e| read_error(&self.io_error_kinds, e, false))?;
        let Some(generation) = Generation::from_meta(&meta) else {
            return self
                .open_checkpoint_file(&loc, Some(meta.size))
                .await
                .map_err(|e| read_error(&self.io_error_kinds, e, true));
        };
        Ok(CheckpointRead {
            size: meta.size,
            source: CheckpointSource::Ranges(loc, generation),
        })
    }

    async fn open_checkpoint_file(
        &self,
        loc: &ResolvedLocation,
        expected_size: Option<u64>,
    ) -> PolarsResult<CheckpointRead> {
        let slot = if loc.is_local {
            self.local_file_slots.try_acquire().ok()
        } else {
            None
        };
        let keep_open = slot.is_some();
        let path = &loc.path;
        let read = loc
            .store
            .exec_with_rebuild_retry_on_err(path, |s| async move {
                // Acquire before GET so concurrent spill reads do not accumulate open files.
                let stage_lock = if keep_open {
                    None
                } else {
                    Some(self.stage_lock.clone().lock_owned().await)
                };
                let result = s.get(path).await?;
                let size = result.meta.size;
                if expected_size.is_some_and(|expected| size != expected) {
                    return Err(object_store::Error::Precondition {
                        path: path.to_string(),
                        source: "object size differs between HEAD and GET".into(),
                    });
                }
                let source = match result.payload {
                    GetResultPayload::File(file, _) if keep_open => {
                        CheckpointSource::File(CheckpointFile::Local(Arc::new(file)))
                    },
                    _ => {
                        let spool = self
                            .checkpoint_spool
                            .get_or_try_init(CheckpointSpool::new)
                            .await?;
                        let offset = spool.stage(result, stage_lock).await?;
                        CheckpointSource::File(CheckpointFile::Staged(spool.clone(), offset))
                    },
                };
                Ok(CheckpointRead { size, source })
            })
            .await?;
        if let Some(slot) = slot
            && matches!(
                read.source,
                CheckpointSource::File(CheckpointFile::Local(_))
            )
        {
            // The file lives in the checkpoint map for this entire storage handle.
            slot.forget();
        }
        Ok(read)
    }
}

/// One anonymous file for all staged checkpoints on a storage handle.
struct CheckpointSpool {
    file: std::fs::File,
    cursor: Mutex<()>,
    end: AtomicU64,
}

fn spool_error(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> object_store::Error {
    object_store::Error::Generic {
        store: "staging Delta checkpoint",
        source: error.into(),
    }
}

impl CheckpointSpool {
    async fn new() -> object_store::Result<Arc<Self>> {
        let file = ASYNC
            .spawn_blocking(|| tempfile::tempfile_in(&*POLARS_TEMP_DIR_BASE_PATH))
            .await
            .map_err(spool_error)?
            .map_err(spool_error)?;
        Ok(Arc::new(Self {
            file,
            cursor: Default::default(),
            end: AtomicU64::new(0),
        }))
    }

    fn copy_at(&self, offset: u64, source: &mut impl Read) -> std::io::Result<u64> {
        let _cursor = self.cursor.lock().unwrap();
        let mut file = &self.file;
        file.seek(SeekFrom::Start(offset))?;
        std::io::copy(source, &mut file)
    }

    fn read_at(&self, offset: u64, len: usize) -> PolarsResult<Bytes> {
        // Windows positional reads move the file cursor.
        #[cfg(windows)]
        let _cursor = self.cursor.lock().unwrap();
        read_buffered(&self.file, offset, len).map(Bytes::from_owner)
    }

    async fn stage(
        self: &Arc<Self>,
        result: GetResult,
        stage_lock: Option<OwnedMutexGuard<()>>,
    ) -> object_store::Result<u64> {
        let size = result.meta.size;
        // Reserve a disjoint interval even if copying is cancelled while a blocking write runs.
        let offset = self
            .end
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |end| {
                end.checked_add(size)
            })
            .map_err(|_| spool_error("checkpoint spool size overflow"))?;
        let written = match result.payload {
            GetResultPayload::File(file, _) => {
                let spool = self.clone();
                ASYNC
                    .spawn_blocking(move || {
                        // Keep the slot until the input file closes, even if the future is dropped.
                        let _stage_lock = stage_lock;
                        let mut source = file.take(size);
                        spool.copy_at(offset, &mut source)
                    })
                    .await
                    .map_err(spool_error)?
                    .map_err(spool_error)?
            },
            _ => {
                let _stage_lock = stage_lock;
                let mut stream = result.into_stream();
                let mut written = 0;
                while let Some(chunk) = stream.try_next().await? {
                    let len = chunk.len() as u64;
                    if len > size - written {
                        return Err(spool_error("checkpoint response exceeded its size"));
                    }
                    let spool = self.clone();
                    ASYNC
                        .spawn_blocking(move || {
                            spool.copy_at(offset + written, &mut chunk.as_ref())
                        })
                        .await
                        .map_err(spool_error)?
                        .map_err(spool_error)?;
                    written += len;
                }
                written
            },
        };
        if written != size {
            return Err(spool_error(
                "checkpoint response length did not match its size",
            ));
        }
        Ok(offset)
    }
}

async fn fetch_bytes(
    loc: &ResolvedLocation,
    generation: Option<&Generation>,
    range: Option<Range<u64>>,
) -> PolarsResult<Bytes> {
    let path = &loc.path;
    let result = loc
        .store
        .exec_with_rebuild_retry_on_err(path, |s| {
            let mut options = GetOptions {
                range: range.clone().map(Into::into),
                ..Default::default()
            };
            match generation {
                Some(Generation::ETag(tag)) => options.if_match = Some(format!("\"{tag}\"")),
                Some(Generation::Version(version)) => options.version = Some(version.clone()),
                None => {},
            }
            async move {
                let result = match s.get_opts(path, options).await {
                    Ok(result) => result,
                    // An absent object (`_last_checkpoint` of a table without a checkpoint)
                    // is an answer: retrying on a rebuilt store costs a TCP and TLS
                    // handshake and a credential refresh per such read.
                    Err(error @ object_store::Error::NotFound { .. }) => return Ok(Err(error)),
                    Err(error) => return Err(error),
                };
                // Some HTTP servers accept conditional requests without enforcing them.
                if generation.is_some_and(|generation| !generation.matches(&result.meta)) {
                    return Err(object_store::Error::Precondition {
                        path: path.to_string(),
                        source: "object generation does not match".into(),
                    });
                }
                result.bytes().await.map(Ok)
            }
        })
        .await?;
    result.map_err(|error| loc.store.error_context(path).attach_err_info(error).into())
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

fn is_last_checkpoint(url: &str) -> bool {
    url.ends_with("/_delta_log/_last_checkpoint")
}

/// The checkpoint version a `_last_checkpoint` names, if it parses.
fn hint_version(bytes: &[u8]) -> Option<i64> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()?
        .get("version")?
        .as_i64()
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
            let generation = storage
                .listed
                .lock()
                .unwrap()
                .get(&url)
                .map(|(generation, _)| generation.clone());
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
                Some(c) if is_last_checkpoint(&url) => {
                    let key = format!("delta-hint:{:?}:{url:?}", c.scope);
                    // The entry is the version the recording plan resolved (8 bytes, big endian),
                    // then the file; only a plan pinned to that version may use it. Empty
                    // after the version is a recorded 404.
                    if let Some(pinned) = storage.pinned_version
                        && let Some(cached) = c.cache.get(&key)
                        && let Some((resolved, hint)) = cached.split_first_chunk::<8>()
                        && i64::from_be_bytes(*resolved) == pinned
                    {
                        let hint = cached.slice_ref(hint);
                        if hint.is_empty() {
                            storage.hint_served.store(true, Ordering::Relaxed);
                            return Err(FfiError::new(
                                FfiErrorKind::NOT_FOUND,
                                format!("not found: {url}"),
                            ));
                        }
                        if hint_version(&hint).is_some_and(|v| v <= pinned) {
                            storage.hint_served.store(true, Ordering::Relaxed);
                            return Ok(FfiBuf::from_owner(hint));
                        }
                    }
                    match fetch().await {
                        Ok(bytes) => {
                            *storage.hint_read.lock().unwrap() = Some((key, bytes.clone()));
                            bytes
                        },
                        Err(e) => {
                            if e.kind() == FfiErrorKind::NOT_FOUND {
                                *storage.hint_read.lock().unwrap() = Some((key, Bytes::new()));
                            }
                            return Err(e);
                        },
                    }
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
                .await?;
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
                Some(CheckpointSource::Ranges(_, generation)) => Some(generation),
                _ => None,
            };
            let range = start..end;
            let fetch = || async {
                match source {
                    Some(CheckpointSource::File(file)) => {
                        let file = file.clone();
                        ASYNC
                            .spawn_blocking(move || file.read_at(start, len))
                            .await
                            .map_err(|e| FfiError::new(FfiErrorKind::OTHER, e.to_string()))?
                            .map_err(|e| storage.io_error_kinds.to_ffi_err(e))
                    },
                    Some(CheckpointSource::Ranges(loc, generation)) => {
                        fetch_bytes(loc, Some(generation), Some(range.clone()))
                            .await
                            .map_err(|e| read_error(&storage.io_error_kinds, e, true))
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
                // A range read without HEAD has no pin and must not populate its cache.
                Some(c) if is_cacheable(&url) && source.is_some() => {
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
                        .exec_with_rebuild_retry_on_err(path, |s| {
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
                        .exec_with_rebuild_retry_on_err(path, |s| async move {
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
                    identities.push((url.clone(), (generation, object.size)));
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
    fn recorded_hint_drops_checkpoint_schemas() {
        let hint = br#"{"version":5,"size":3,"checkpointSchema":{"type":"struct"},"sidecarFileSchema":{},"parts":2}"#;
        let (stripped, version) = strip_checkpoint_schemas(hint).unwrap();
        assert_eq!(version, 5);
        let value: serde_json::Value = serde_json::from_slice(&stripped).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"version": 5, "size": 3, "parts": 2})
        );
        assert!(strip_checkpoint_schemas(b"not json").is_none());
        assert!(strip_checkpoint_schemas(b"[]").is_none());
    }

    #[test]
    fn last_checkpoint_version_is_read_from_the_hint() {
        assert_eq!(hint_version(br#"{"version":7,"size":3}"#), Some(7));
        assert_eq!(hint_version(b"{}"), None);
        assert_eq!(hint_version(b"not json"), None);
        assert!(is_last_checkpoint("s3://b/t/_delta_log/_last_checkpoint"));
        assert!(!is_last_checkpoint("s3://b/t/_delta_log/1.json"));
    }

    #[test]
    fn checkpoint_etag_accepts_list_quoting_and_rejects_weak_or_malformed_tags() {
        let mut meta = object_store::ObjectMeta {
            location: "checkpoint.parquet".into(),
            last_modified: chrono::DateTime::UNIX_EPOCH,
            size: 8,
            e_tag: Some("old".into()),
            version: None,
        };
        let listed = Generation::from_meta(&meta).unwrap();
        meta.e_tag = Some("\"old\"".into());
        assert!(listed.matches(&meta));
        let head = Generation::from_meta(&meta).unwrap();
        meta.e_tag = Some("old".into());
        assert!(head.matches(&meta));
        meta.e_tag = Some("\"new\"".into());
        assert!(!head.matches(&meta));
        meta.e_tag = None;
        assert!(!head.matches(&meta));

        for malformed in ["W/\"old\"", "\"old", "old\"", "old tag", "", "*", "old,new"] {
            meta.e_tag = Some(malformed.into());
            assert!(Generation::from_meta(&meta).is_none(), "{malformed:?}");
            assert!(!listed.matches(&meta), "{malformed:?}");
        }
        meta.e_tag = Some("W/\"old\"".into());
        meta.version = Some("version".into());
        assert!(matches!(
            Generation::from_meta(&meta),
            Some(Generation::Version(version)) if version == "version"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn local_checkpoints_share_a_spool_after_the_open_file_budget() {
        let table = tempfile::tempdir().unwrap();
        let sidecars = table.path().join("_delta_log").join("_sidecars");
        std::fs::create_dir_all(&sidecars).unwrap();
        let files = [
            ("first.parquet", b"abcdefgh"),
            ("second.parquet", b"ijklmnop"),
            ("third.parquet", b"qrstuvwx"),
        ];
        let ctx = HostCtx {
            cloud_options: None,
            metadata_cache: None,
            io_error_kinds: Default::default(),
            pinned_version: None,
            hint_served: Default::default(),
            hint_read: Default::default(),
        };
        ASYNC.block_on(async {
            let handle = unsafe {
                host_get_storage(ctx.host().ctx, FfiStr::new(table.path().to_str().unwrap()))
            }
            .await
            .into_result()
            .unwrap();
            let mut storage = unsafe { Arc::from_raw(handle.0.cast::<HostStorage>()) };
            Arc::get_mut(&mut storage).unwrap().local_file_slots = Semaphore::new(1);

            for (index, (name, data)) in files.iter().enumerate() {
                let path = sidecars.join(name);
                std::fs::write(&path, data).unwrap();
                let url = FfiStr::new(path.to_str().unwrap());
                let size = unsafe { storage_head(handle, url) }
                    .await
                    .into_result()
                    .unwrap();
                assert_eq!(size, 8);

                let checkpoints = storage.checkpoints.lock().unwrap();
                let read = checkpoints[path.to_str().unwrap()].get().unwrap();
                if index == 0 {
                    assert!(matches!(
                        read.source,
                        CheckpointSource::File(CheckpointFile::Local(_))
                    ));
                } else {
                    let CheckpointSource::File(CheckpointFile::Staged(spool, offset)) =
                        &read.source
                    else {
                        panic!("overflow checkpoint retained a separate file descriptor");
                    };
                    assert!(Arc::ptr_eq(spool, storage.checkpoint_spool.get().unwrap()));
                    assert_eq!(*offset, (index as u64 - 1) * size);
                }
            }

            for (name, data) in files {
                let path = sidecars.join(name);
                let replacement = sidecars.join("replacement");
                std::fs::write(&replacement, b"12345678").unwrap();
                std::fs::rename(&replacement, &path).unwrap();
                let url = FfiStr::new(path.to_str().unwrap());
                let range = unsafe { storage_get_range(handle, url, 2, 6) }
                    .await
                    .into_result()
                    .unwrap();
                assert_eq!(range.as_slice(), &data[2..6]);
            }
        });
    }

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
        ASYNC.block_on(async {
            let spool = CheckpointSpool::new().await.unwrap();
            let offset = spool.stage(result, None).await.unwrap();
            assert_eq!(spool.read_at(offset + 2, 4).unwrap().as_ref(), b"cdef");
        });
    }
}
