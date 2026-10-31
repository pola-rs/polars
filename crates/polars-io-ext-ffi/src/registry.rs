//! Finding the contract structs of installed plugins, for hosts that call plugins from anywhere
//! in a query (not only from where a plugin was handed over, as `iceberg.v1` is).
//!
//! A plugin is named by its Python package (e.g. `polars_database`), which is what query plans
//! record. The host installs a [`Locator`] once (Polars' Python module installs one that imports
//! the package and uses the handshake of the crate docs); lookups are cached for the life of the
//! process, since plugins are never unloaded.
use std::ffi::{CStr, c_void};
use std::sync::{Mutex, OnceLock};

/// Returns the pointer of the contract struct of plugin ID `id` in `package`, or an error message
/// (e.g. the package is not installed, or does not provide the ID).
pub type Locator = Box<dyn Fn(&str, &CStr) -> Result<*const c_void, String> + Send + Sync>;

static LOCATOR: OnceLock<Locator> = OnceLock::new();

#[allow(clippy::type_complexity)]
static CACHE: Mutex<Vec<(String, &'static CStr, usize)>> = Mutex::new(Vec::new());

/// Installs the locator. Returns `false` if one was installed already (which is kept).
pub fn set_locator(locator: Locator) -> bool {
    LOCATOR.set(locator).is_ok()
}

/// The contract struct of `id` in `package`.
///
/// # Safety
/// `T` must be the `Plugin` struct of the contract `id`.
pub unsafe fn lookup<T>(package: &str, id: &'static CStr) -> Result<&'static T, String> {
    let cached = |cache: &[(String, &'static CStr, usize)]| {
        cache
            .iter()
            .find(|(p, i, _)| p == package && *i == id)
            .map(|(_, _, ptr)| *ptr)
    };
    if let Some(ptr) = cached(&CACHE.lock().unwrap()) {
        return Ok(unsafe { &*(ptr as *const T) });
    }
    let Some(locator) = LOCATOR.get() else {
        return Err(format!(
            "cannot load the I/O plugin {package}: this Polars build cannot load I/O plugins"
        ));
    };
    // Without the cache lock: the locator may wait for other threads (e.g. for the GIL).
    let ptr = locator(package, id)?;
    if ptr.is_null() {
        return Err(format!(
            "I/O plugin {package} returned a null {id:?} contract"
        ));
    }
    let mut cache = CACHE.lock().unwrap();
    if cached(&cache).is_none() {
        cache.push((package.to_owned(), id, ptr as usize));
    }
    // SAFETY: plugins are never unloaded; the caller guarantees the type.
    Ok(unsafe { &*(ptr as *const T) })
}
