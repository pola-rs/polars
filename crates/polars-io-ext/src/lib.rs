//! Native I/O plugins for Polars, loaded through the contracts in `polars_io_ext_ffi`.
//!
//! Each format lives in its own module behind a Cargo feature of the same name:
//! * [`iceberg`] (feature `iceberg`): Iceberg scan planning, contract `polars.io_plugin.iceberg.v1`.
//!
//! A build without any format feature compiles to a plugin that exports no plugin IDs.
use std::any::Any;
use std::panic::AssertUnwindSafe;

#[allow(unused_imports)]
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};

#[cfg(feature = "iceberg")]
pub mod iceberg;

#[cfg(feature = "python")]
mod python;

/// Python distribution name, as used in pip requirements and error messages. Each format is
/// released as its own wheel built from this crate with only that format's feature enabled.
#[cfg(feature = "iceberg")]
pub const PACKAGE_NAME: &str = "polars_iceberg";
#[cfg(not(feature = "iceberg"))]
pub const PACKAGE_NAME: &str = "polars_io_ext";

/// Unwinding across `extern "C"` is undefined behavior, so panics become errors.
#[allow(dead_code)]
pub(crate) fn catch_panic<T>(f: impl FnOnce() -> Result<T, FfiError>) -> Result<T, FfiError> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        Err(FfiError::new(
            FfiErrorKind::PANIC,
            format!("{PACKAGE_NAME} panicked: {}", panic_message(&*payload)),
        ))
    })
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "<non-string panic payload>"
    }
}
