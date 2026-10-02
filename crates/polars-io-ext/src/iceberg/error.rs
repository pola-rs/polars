//! Errors returned to the host. They cross the ABI as `FfiError`, whose kind selects the Python
//! exception type raised by Polars.
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};

pub type IcebergResult<T> = Result<T, FfiError>;

/// Malformed table metadata or manifests.
pub fn err_invalid_data(msg: impl Into<String>) -> FfiError {
    FfiError::new(FfiErrorKind::OTHER, format!("iceberg: {}", msg.into()))
}

/// A table feature this plugin does not support. Never falls back to another planner.
pub fn err_not_implemented(msg: impl Into<String>) -> FfiError {
    FfiError::new(
        FfiErrorKind::NOT_IMPLEMENTED,
        format!("iceberg: unsupported: {}", msg.into()),
    )
}

/// An invalid user parameter (e.g. an unknown snapshot ID). Raised as `ValueError` in Python.
pub fn err_invalid_input(msg: impl Into<String>) -> FfiError {
    FfiError::new(FfiErrorKind::INVALID_INPUT, msg.into())
}
