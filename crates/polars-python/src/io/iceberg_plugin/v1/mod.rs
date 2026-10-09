//! Host of the contract `polars.io_plugin.iceberg.v1`.
mod convert;
mod host;

use std::sync::atomic::Ordering;

use polars::prelude::{CastColumnsPolicy, DslPlan};
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind};
use polars_io_ext_ffi::iceberg_v1::{Plugin, Request};

use self::host::HostCtx;
use super::metadata_cache::ScopedMetadataCache;

/// Plan the scan with the plugin. Blocks; must be called without the GIL.
pub(super) fn scan(
    plugin: &Plugin,
    plugin_name: &str,
    request: &Request,
    cloud_options: Option<CloudOptions>,
    metadata_cache: Option<ScopedMetadataCache>,
    cast_columns_policy: CastColumnsPolicy,
) -> PolarsResult<DslPlan> {
    let ctx = HostCtx {
        cloud_options: cloud_options.clone(),
        metadata_cache,
        io_error_kinds: Default::default(),
    };
    let host = ctx.host();

    // SAFETY: `host` and `ctx` outlive the call.
    let output = request
        .with_ffi(|request| unsafe { (plugin.plan)(&host, request) })
        .into_result()
        .map_err(|e| {
            let io_kind = ctx.io_error_kinds.kind_of(&e.message());
            ffi_to_polars_err(e, plugin_name, io_kind)
        })?;

    if request.verbose
        && let Some(c) = &ctx.metadata_cache
    {
        // Same message as the PyIceberg planner.
        eprintln!(
            "IcebergScanResolver: to_dataset_scan(): metadata file cache: hits: {}, misses: {}, \
            cached bytes: {}",
            c.stats.hits.load(Ordering::Relaxed),
            c.stats.misses.load(Ordering::Relaxed),
            c.cache.total_bytes(),
        );
    }

    let scan = convert::import_output(output)?;
    convert::build_scan(scan, cloud_options, cast_columns_policy)
}

fn ffi_to_polars_err(
    e: FfiError,
    plugin: &str,
    io_kind: Option<std::io::ErrorKind>,
) -> PolarsError {
    if e.kind() == FfiErrorKind::INVALID_INPUT {
        // Invalid user input (e.g. an unknown snapshot ID) is reported as a `ValueError` with the
        // plugin's message, like the parameter errors of the PyIceberg planner.
        return pyo3::exceptions::PyValueError::new_err(e.message()).into();
    }
    if e.kind() == FfiErrorKind::NOT_IMPLEMENTED {
        // Unsupported table features (e.g. equality deletes) are raised as `NotImplementedError`,
        // on which `IcebergScanResolver` falls back to PyIceberg unless the plugin is required.
        return pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "{plugin}: {}",
            e.message()
        ))
        .into();
    }
    let io_kind = match e.kind() {
        FfiErrorKind::NOT_FOUND => Some(std::io::ErrorKind::NotFound),
        // Storage errors keep their Python exception type (e.g. `PermissionError`), as with the
        // PyIceberg planner.
        FfiErrorKind::IO => Some(
            io_kind
                .filter(|k| *k != std::io::ErrorKind::NotFound)
                .unwrap_or(std::io::ErrorKind::Other),
        ),
        _ => None,
    };
    if let Some(kind) = io_kind {
        return std::io::Error::new(kind, format!("{plugin}: {}", e.message())).into();
    }
    polars_err!(
        ComputeError: "{} failed ({}): {}",
        plugin, e.kind().name(), e.message()
    )
}
