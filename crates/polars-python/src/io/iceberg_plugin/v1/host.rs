//! Host functions of `iceberg.v1`: storage over `PolarsObjectStore`, executed on the Polars IO
//! runtime.
use std::ffi::c_void;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use object_store::ObjectStoreExt;
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult};
use polars_io::cloud::concurrency_config::ConcurrencyStrategy;
use polars_io::cloud::{
    CloudLocation, CloudOptions, PolarsObjectStore, build_object_store, object_path_from_str,
};
use polars_io_ext_ffi::common::{
    FfiBuf, FfiError, FfiErrorKind, FfiFuture, FfiResult, FfiStr, FutureExt,
};
use polars_io_ext_ffi::iceberg_v1::{Host, StorageHandle};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::pl_path::PlRefPath;

use crate::io::iceberg_plugin::metadata_cache::{ScopedMetadataCache, is_cacheable};

/// State shared with the plugin for the duration of one `plan` call; [`Host::ctx`] points to it.
pub(super) struct HostCtx {
    /// `storage_options` and credential provider of the scan.
    pub cloud_options: Option<CloudOptions>,
    pub metadata_cache: Option<ScopedMetadataCache>,
    pub last_io_error: LastIoError,
}

/// Kind of the last IO error of the scan's storage calls. The contract only distinguishes
/// `NOT_FOUND` and `IO`; this restores e.g. `PermissionDenied` when the plugin fails with it.
#[derive(Clone, Default)]
pub(super) struct LastIoError(Arc<Mutex<Option<std::io::ErrorKind>>>);

impl LastIoError {
    pub fn get(&self) -> Option<std::io::ErrorKind> {
        *self.0.lock().unwrap()
    }

    fn to_ffi_err(&self, e: PolarsError) -> FfiError {
        if let PolarsError::IO { error, .. } = &e {
            *self.0.lock().unwrap() = Some(error.kind());
        }
        polars_to_ffi_err(e)
    }
}

impl HostCtx {
    /// The host functions. `self` must outlive the plugin call.
    pub fn host(&self) -> Host {
        Host {
            ctx: self as *const Self as *const c_void,
            get_storage: host_get_storage,
            storage_get,
            storage_head,
            storage_release,
            log: host_log,
        }
    }
}

fn polars_to_ffi_err(e: PolarsError) -> FfiError {
    let kind = match &e {
        PolarsError::IO { error, .. } if error.kind() == std::io::ErrorKind::NotFound => {
            FfiErrorKind::NOT_FOUND
        },
        PolarsError::IO { .. } => FfiErrorKind::IO,
        _ => FfiErrorKind::OTHER,
    };
    FfiError::new(kind, e.to_string())
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
                last_io_error: ctx.last_io_error.clone(),
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
    last_io_error: LastIoError,
}

struct ResolvedLocation {
    store: PolarsObjectStore,
    path: object_store::path::Path,
}

impl HostStorage {
    async fn resolve(&self, url: &str) -> PolarsResult<ResolvedLocation> {
        let (CloudLocation { prefix, .. }, store) =
            build_object_store(PlRefPath::new(url), self.cloud_options.as_ref(), false).await?;

        Ok(ResolvedLocation {
            store,
            path: object_path_from_str(&prefix)?,
        })
    }
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
                let to_ffi_err = |e| storage.last_io_error.to_ffi_err(e);
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
                    let key = format!("{}:{url}", c.scope);
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
            let to_ffi_err = |e| storage.last_io_error.to_ffi_err(e);
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
