//! Host of the contract `polars.io_plugin.iceberg.v1`.
mod convert;
mod host;

use polars::prelude::{CastColumnsPolicy, DslPlan};
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
use polars_io_ext_ffi::iceberg_v1::{Plugin, Request};

use self::host::HostCtx;

/// Plan the scan with the plugin. Blocks; must be called without the GIL.
pub(super) fn scan(
    plugin: &Plugin,
    plugin_name: &str,
    request: &Request,
    cloud_options: Option<CloudOptions>,
    cast_columns_policy: CastColumnsPolicy,
) -> PolarsResult<DslPlan> {
    let ctx = HostCtx {
        cloud_options: cloud_options.clone(),
    };
    let host = ctx.host();

    // SAFETY: `host` and `ctx` outlive the call.
    let output = request
        .with_ffi(|request| unsafe { (plugin.plan)(&host, request) })
        .into_result()
        .map_err(|e| ffi_to_polars_err(e, plugin_name))?;

    let scan = convert::import_output(output)?;
    convert::build_scan(scan, cloud_options, cast_columns_policy)
}

fn ffi_to_polars_err(e: FfiError, plugin: &str) -> PolarsError {
    if e.kind() == FfiErrorKind::INVALID_INPUT {
        // Invalid user input (e.g. an unknown snapshot ID) is reported as a `ValueError` with the
        // plugin's message, like the parameter errors of the PyIceberg planner.
        return pyo3::exceptions::PyValueError::new_err(e.message()).into();
    }
    polars_err!(
        ComputeError: "{} failed ({}): {}",
        plugin, e.kind().name(), e.message()
    )
}
