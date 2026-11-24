//! The `iceberg.v1` request.
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
use polars_io_ext_ffi::iceberg_v1::FfiRequest;
pub use polars_io_ext_ffi::iceberg_v1::Request;

use crate::iceberg::host::PluginResult;
use crate::iceberg::planner::SnapshotSelection;

/// # Safety
/// `request` must point to a valid request for the duration of the call.
pub unsafe fn parse(request: *const FfiRequest) -> PluginResult<Request> {
    let invalid = |msg: &str| FfiError::new(FfiErrorKind::INVALID_ARGUMENT, msg);
    let request = unsafe { request.as_ref() }.ok_or_else(|| invalid("request is null"))?;
    Request::from_ffi(request).map_err(|e| invalid(&format!("request string is not UTF-8: {e}")))
}

pub fn selection(request: &Request) -> SnapshotSelection {
    SnapshotSelection {
        snapshot_id: request.snapshot_id,
        from_snapshot_id_exclusive: request.from_snapshot_id_exclusive,
        to_snapshot_id_inclusive: request.to_snapshot_id_inclusive,
    }
}
