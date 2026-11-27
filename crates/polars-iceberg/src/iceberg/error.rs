//! Errors returned to the host. They cross the ABI as `FfiError`, whose kind selects the Python
//! exception type raised by Polars.
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};

pub type IcebergResult<T> = Result<T, FfiError>;

/// Malformed table metadata or manifests.
pub fn err_invalid_data(msg: impl Into<String>) -> FfiError {
    FfiError::new(FfiErrorKind::OTHER, format!("iceberg: {}", msg.into()))
}

/// A table feature this plugin does not support. Raised as `NotImplementedError` in Python, on
/// which Polars plans the scan with PyIceberg unless `POLARS_ICEBERG_PLANNER=plugin` is set.
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

/// Prefix an error's message with `context`, keeping its kind (so that e.g. an unsupported feature
/// found while reading a manifest still lets Polars fall back to PyIceberg).
pub fn with_context(e: FfiError, context: impl std::fmt::Display) -> FfiError {
    let msg = e.message();
    let msg = msg.strip_prefix("iceberg: ").unwrap_or(&msg);
    FfiError::new(e.kind(), format!("iceberg: {context}: {msg}"))
}
