use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
pub type Result<T> = std::result::Result<T, FfiError>;
pub fn invalid(msg: impl std::fmt::Display) -> FfiError {
    FfiError::new(FfiErrorKind::OTHER, format!("delta: {msg}"))
}
pub fn input(msg: impl std::fmt::Display) -> FfiError {
    FfiError::new(FfiErrorKind::INVALID_INPUT, msg.to_string())
}
pub fn unsupported(msg: impl std::fmt::Display) -> FfiError {
    FfiError::new(
        FfiErrorKind::NOT_IMPLEMENTED,
        format!("delta: unsupported {msg}"),
    )
}
