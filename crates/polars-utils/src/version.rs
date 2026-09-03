use std::sync::{Arc, LazyLock, Mutex};

static POLARS_NAME: LazyLock<Mutex<Arc<String>>> =
    LazyLock::new(|| Mutex::new(Arc::new(String::from("Polars (standalone)"))));

static POLARS_VERSION: LazyLock<Mutex<Arc<String>>> =
    LazyLock::new(|| Mutex::new(Arc::new(String::from(env!("CARGO_PKG_VERSION")))));

static POLARS_BUILD: LazyLock<Mutex<Arc<String>>> =
    LazyLock::new(|| Mutex::new(Arc::new(String::from("<unknown>"))));

/// Set the name Polars uses for e.g. file metadata.
pub fn set_polars_lib_name(name: &str) {
    *POLARS_NAME.lock().unwrap() = Arc::new(name.into());
}

/// Set the version Polars uses for e.g. file metadata.
pub fn set_polars_lib_version(version: &str) {
    *POLARS_VERSION.lock().unwrap() = Arc::new(version.into());
}

/// Set the build Polars uses for e.g. file metadata.
pub fn set_polars_build_version(build: &str) {
    *POLARS_BUILD.lock().unwrap() = Arc::new(build.into());
}

/// Get the name Polars uses for e.g. file metadata.
pub fn get_polars_lib_name() -> Arc<String> {
    POLARS_NAME.lock().unwrap().clone()
}

/// Get the version Polars uses for e.g. file metadata.
pub fn get_polars_lib_version() -> Arc<String> {
    POLARS_VERSION.lock().unwrap().clone()
}

/// Get the build Polars uses for e.g. file metadata.
pub fn get_polars_build_version() -> Arc<String> {
    POLARS_BUILD.lock().unwrap().clone()
}
