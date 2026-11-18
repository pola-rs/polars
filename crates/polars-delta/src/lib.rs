//! Delta Lake deletion-vector decoding and optional native snapshot planning.

#[cfg(feature = "planner")]
use std::panic::AssertUnwindSafe;

#[cfg(feature = "planner")]
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};

pub mod codec;
#[cfg(feature = "planner")]
pub mod delta;
#[cfg(feature = "planner")]
mod ffi;
#[cfg(feature = "python")]
pub mod python;

pub const PACKAGE_NAME: &str = "polars_delta";

#[cfg(feature = "planner")]
pub(crate) fn catch_panic<T>(
    what: &'static str,
    f: impl FnOnce() -> Result<T, FfiError>,
) -> Result<T, FfiError> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(FfiError::new(
            FfiErrorKind::PANIC,
            format!("polars_delta panicked in {what}"),
        ))
    })
}
