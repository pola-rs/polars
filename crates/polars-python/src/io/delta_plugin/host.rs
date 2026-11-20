//! Host functions of `delta.v1`: storage over `PolarsObjectStore`, executed on the Polars IO
//! runtime.
use std::ffi::c_void;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use object_store::ObjectStoreExt;
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::cloud::concurrency_config::ConcurrencyStrategy;
use polars_io::cloud::{
    CloudLocation, CloudOptions, PolarsObjectStore, build_object_store, object_path_from_str,
};
use polars_io_ext_ffi::common::{
    FfiBuf, FfiError, FfiErrorKind, FfiFuture, FfiOption, FfiResult, FfiStr, FfiVec, FutureExt,
};
use polars_io_ext_ffi::delta_v1::{Host, ObjectMeta, StorageHandle};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::pl_path::{CloudScheme, PlRefPath};

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

            let storage = Arc::new(HostStorage {
                cloud_options: ctx.cloud_options.clone(),
                metadata_cache: ctx.metadata_cache.clone(),
                io_error_kinds: ctx.io_error_kinds.clone(),
                listed: Default::default(),
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
    /// Size and modification time of the log files this handle listed. A cached copy is keyed by
    /// them as well, so a table recreated at the same path never serves the old table's files.
    listed: Mutex<std::collections::HashMap<String, (u64, i64)>>,
}

struct ResolvedLocation {
    store: PolarsObjectStore,
    path: object_store::path::Path,
    is_local: bool,
}

impl HostStorage {
    /// The cache key of `url`: its scope and URL, with the identity of the listed object. A file
    /// this handle did not list (a sidecar, named by a UUID) is immutable by name.
    fn cache_key(&self, scope: impl std::fmt::Display, url: &str, suffix: &str) -> String {
        match self.listed.lock().unwrap().get(url) {
            Some((size, modified)) => format!("{scope}:{url}:{size}:{modified}{suffix}"),
            None => format!("{scope}:{url}{suffix}"),
        }
    }
}

impl HostStorage {
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
    // Commits/checkpoints/sidecars are immutable; never cache mutable hints or listings.
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
            let fetch = || async {
                let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
                let loc = storage.resolve(&url).await.map_err(to_ffi_err)?;
                let path = &loc.path;

                loc.store
                    .exec_with_rebuild_retry_on_err(
                        |s| async move { s.get(path).await?.bytes().await },
                    )
                    .await
                    .map_err(to_ffi_err)
            };

            let bytes = match &storage.metadata_cache {
                Some(c) if is_cacheable(&url) => {
                    let key = storage.cache_key(&c.scope, &url, "");
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
            let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
            let loc = storage.resolve(&url?).await.map_err(to_ffi_err)?;

            let meta = loc
                .store
                .head(&loc.path, ConcurrencyStrategy::BytesBased)
                .await
                .map_err(to_ffi_err)?;

            Ok(meta.size)
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
            if end < start {
                return Err(FfiError::new(
                    FfiErrorKind::INVALID_ARGUMENT,
                    "invalid byte range",
                ));
            }
            let range = usize::try_from(start)
                .ok()
                .zip(usize::try_from(end).ok())
                .ok_or_else(|| {
                    FfiError::new(
                        FfiErrorKind::INVALID_ARGUMENT,
                        "range exceeds address space",
                    )
                })?;
            let to_ffi_err = |e| storage.io_error_kinds.to_ffi_err(e);
            let url = url?;
            let loc = storage.resolve(&url).await.map_err(to_ffi_err)?;
            let path = &loc.path;
            let fetch = || async {
                loc.store
                    .exec_with_rebuild_retry_on_err(|s| async move {
                        s.get_range(path, start..end).await
                    })
                    .await
                    .map_err(to_ffi_err)
            };
            let bytes = match &storage.metadata_cache {
                Some(c) if is_cacheable(&url) => {
                    let key = storage.cache_key(&c.scope, &url, &format!(":range:{start}:{end}"));
                    c.cache.get_or_fetch(&key, &c.stats, fetch).await?
                },
                _ => fetch().await?,
            };
            if bytes.len() != range.1 - range.0 {
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
                    let metas = loc
                        .store
                        .exec_with_rebuild_retry_on_err(|s| {
                            let offset = offset.clone();
                            async move {
                                use futures::TryStreamExt;
                                s.list_with_offset(Some(path), &offset)
                                    .try_collect::<Vec<_>>()
                                    .await
                            }
                        })
                        .await
                        .map_err(to_ffi_err)?;
                    metas
                        .into_iter()
                        .filter(|m| {
                            m.location
                                .prefix_match(path)
                                .is_some_and(|mut parts| parts.nth(1).is_none())
                        })
                        .collect::<Vec<_>>()
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
                identities.push((url.clone(), (object.size, modified)));
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
    dir.child(name)
}
