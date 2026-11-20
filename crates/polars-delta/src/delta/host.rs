//! Safe wrappers around the host functions of `delta.v1`.
use std::sync::Arc;

use polars_io_ext_ffi::common::{FfiBuf, FfiError, FfiErrorKind, FfiStr};
use polars_io_ext_ffi::delta_v1::{self, StorageHandle};

pub type PluginResult<T> = Result<T, FfiError>;

/// Access to the host for the duration of one `plan` call.
#[derive(Clone, Copy)]
pub struct Host {
    vtable: &'static delta_v1::Host,
    verbose: bool,
}

impl Host {
    /// # Safety
    /// `vtable` must be the pointer passed to the current `plan` call, and the returned `Host`
    /// (and every [`Storage`] obtained from it) must not be used after that call returns.
    pub unsafe fn from_raw(vtable: *const delta_v1::Host, verbose: bool) -> PluginResult<Self> {
        if vtable.is_null() {
            return Err(FfiError::new(
                FfiErrorKind::INVALID_ARGUMENT,
                "host passed a null host struct",
            ));
        }
        Ok(Self {
            vtable: unsafe { &*vtable },
            verbose,
        })
    }

    pub async fn get_storage(&self, url: &str) -> PluginResult<Storage> {
        let url = normalize_path(url);
        let fut = unsafe { (self.vtable.get_storage)(self.vtable.ctx, FfiStr::new(&url)) };
        let handle = fut.await.into_result()?;
        Ok(Storage(Arc::new(StorageInner {
            host: *self,
            handle,
        })))
    }

    /// Log a message, printed only when the host is verbose.
    pub fn debug(&self, msg: &str) {
        if self.verbose {
            unsafe { (self.vtable.log)(self.vtable.ctx, FfiStr::new(msg)) }
        }
    }
}

struct StorageInner {
    host: Host,
    handle: StorageHandle,
}

impl Drop for StorageInner {
    fn drop(&mut self) {
        unsafe { (self.host.vtable.storage_release)(self.handle) }
    }
}

/// Host storage, shared by the tasks of one `plan` call.
#[derive(Clone)]
pub struct Storage(Arc<StorageInner>);

impl Storage {
    pub async fn get(&self, url: &str) -> PluginResult<FfiBuf> {
        let url = normalize_path(url);
        let fut = unsafe { (self.0.host.vtable.storage_get)(self.0.handle, FfiStr::new(&url)) };
        fut.await.into_result()
    }

    /// Size of the object in bytes.
    pub async fn head(&self, url: &str) -> PluginResult<u64> {
        let url = normalize_path(url);
        let fut = unsafe { (self.0.host.vtable.storage_head)(self.0.handle, FfiStr::new(&url)) };
        fut.await.into_result()
    }
}

/// PyIceberg on Windows uses `file://C:/` rather than `file:///C:/`.
pub fn normalize_path(path: &str) -> String {
    // `file://C:/x` (as written by PyIceberg on Windows) has the drive letter as its
    // authority; other authorities (UNC hosts, `localhost`) are kept.
    match path.strip_prefix("file://") {
        Some(rest) if matches!(rest.as_bytes(), [d, b':', ..] if d.is_ascii_alphabetic()) => {
            format!("file:///{rest}")
        },
        _ => path.to_owned(),
    }
}

#[derive(Debug, Clone)]
pub struct Object {
    pub url: String,
    pub modified_ms: Option<i64>,
}
impl Storage {
    pub async fn list(&self, prefix: &str) -> PluginResult<Vec<Object>> {
        self.list_after(prefix, None).await
    }
    pub async fn list_after(&self, prefix: &str, start: Option<&str>) -> PluginResult<Vec<Object>> {
        let entries = unsafe {
            (self.0.host.vtable.storage_list)(
                self.0.handle,
                FfiStr::new(prefix),
                start.map(FfiStr::new).into(),
            )
        }
        .await
        .into_result()?;
        entries
            .as_slice()
            .iter()
            .map(|e| {
                Ok(Object {
                    url: e.url.as_slice().pipe_str()?,
                    modified_ms: e.modified_ms.into_option(),
                })
            })
            .collect()
    }
    pub async fn range(&self, url: &str, start: u64, end: u64) -> PluginResult<FfiBuf> {
        if end < start {
            return Err(super::error::invalid("invalid range"));
        }
        let bytes = unsafe {
            (self.0.host.vtable.storage_get_range)(self.0.handle, FfiStr::new(url), start, end)
        }
        .await
        .into_result()?;
        if bytes.as_slice().len() as u64 != end - start {
            return Err(super::error::invalid("short storage range"));
        }
        Ok(bytes)
    }
}
trait BytesString {
    fn pipe_str(&self) -> PluginResult<String>;
}
impl BytesString for [u8] {
    fn pipe_str(&self) -> PluginResult<String> {
        std::str::from_utf8(self)
            .map(str::to_owned)
            .map_err(super::error::invalid)
    }
}
