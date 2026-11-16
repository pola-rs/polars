//! Native Iceberg scan planning plugin for Polars, released as the Python package
//! `polars-iceberg`.
//!
//! Polars loads it through the contract `polars.io_plugin.iceberg.v1` of `polars_io_ext_ffi`; the
//! planner lives in [`iceberg`]. With the `python` feature, [`python`] builds the extension
//! module that exposes the contract as a PyCapsule.
use std::any::Any;
use std::panic::AssertUnwindSafe;

use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};

pub mod iceberg;

#[cfg(feature = "python")]
pub mod python;

/// Python distribution name, as used in pip requirements and error messages.
pub const PACKAGE_NAME: &str = "polars_iceberg";

/// Unwinding across `extern "C"` is undefined behavior, so panics become errors.
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
