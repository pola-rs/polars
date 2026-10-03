//! Host side of the Iceberg I/O plugin (`polars_iceberg`), see `polars_io_ext_ffi`.
//!
//! `IcebergScanResolver` (Python) obtains the capsule of the newest plugin ID shared with
//! [`SUPPORTED_IDS`] and calls [`_iceberg_plugin_scan`]. Dispatch is by exact capsule name; any
//! other name is refused.
mod v1;

use polars::prelude::{CastColumnsPolicy, CloudScheme};
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io_ext_ffi::{PluginId, iceberg_v1};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use crate::error::PyPolarsErr;
use crate::io::cloud_options::OptPyCloudOptions;
use crate::lazyframe::PyLazyFrame;
use crate::prelude::Wrap;
use crate::utils::EnterPolarsExt;

/// Iceberg plugin IDs this Polars can call, oldest first. Removing an ID drops support for
/// plugins that only provide it.
pub const SUPPORTED_IDS: &[PluginId] = &[PluginId::IcebergV1];

/// Python distribution name of the plugin, for error messages.
const PLUGIN_NAME: &str = "polars_iceberg";

enum Dispatch {
    V1(&'static iceberg_v1::Plugin),
}

fn dispatch(capsule: &Bound<'_, PyCapsule>) -> PolarsResult<Dispatch> {
    // SAFETY: Plugin capsule names are static; the name is only used during this call.
    let name = capsule
        .name()
        .map_err(|e| polars_err!(ComputeError: "invalid {PLUGIN_NAME} capsule: {e}"))?
        .map(|n| unsafe { n.as_cstr() }.to_owned());

    let id = name
        .as_deref()
        .and_then(PluginId::from_name)
        .filter(|id| SUPPORTED_IDS.contains(id));

    match id {
        Some(id @ PluginId::IcebergV1) => {
            let ptr = capsule
                .pointer_checked(Some(id.name()))
                .map_err(|e| polars_err!(ComputeError: "invalid {PLUGIN_NAME} capsule: {e}"))?;
            // SAFETY: A capsule with this name points to a static `iceberg_v1::Plugin`, and plugin
            // extension modules are never unloaded.
            Ok(Dispatch::V1(unsafe {
                &*(ptr.as_ptr() as *const iceberg_v1::Plugin)
            }))
        },
        None => polars_bail!(
            ComputeError:
            "unsupported {PLUGIN_NAME} plugin ID {}; this Polars supports {:?}",
            name.map_or_else(|| "<none>".to_owned(), |n| format!("{:?}", n.to_string_lossy())),
            SUPPORTED_IDS.iter().map(|id| id.as_str()).collect::<Vec<_>>()
        ),
    }
}

/// Plan an Iceberg scan with the plugin and return it as a parquet scan.
///
/// The keyword arguments up to `testing_fail` are the plugin request (see
/// `polars_io_ext_ffi::iceberg_v1::Request`); `max_threads` is set by the host.
#[pyfunction]
#[pyo3(signature = (
    capsule, *, metadata_location, snapshot_id, from_snapshot_id_exclusive,
    to_snapshot_id_inclusive, projection, filter_columns, row_filter, limit,
    use_metadata_statistics, fast_deletion_count, verbose, testing_fail,
    source_url, storage_options, credential_provider, cast_options
))]
#[allow(clippy::too_many_arguments)]
pub fn _iceberg_plugin_scan(
    py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
    metadata_location: String,
    snapshot_id: Option<i64>,
    from_snapshot_id_exclusive: Option<i64>,
    to_snapshot_id_inclusive: Option<i64>,
    projection: Option<Vec<String>>,
    filter_columns: Option<Vec<String>>,
    row_filter: Option<String>,
    limit: Option<u64>,
    use_metadata_statistics: bool,
    fast_deletion_count: bool,
    verbose: bool,
    testing_fail: Option<String>,
    source_url: &str,
    storage_options: OptPyCloudOptions,
    credential_provider: Option<Py<PyAny>>,
    cast_options: Wrap<CastColumnsPolicy>,
) -> PyResult<PyLazyFrame> {
    let capsule = capsule.cast::<PyCapsule>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{PLUGIN_NAME}: expected a PyCapsule, got {}",
            capsule
                .get_type()
                .name()
                .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string())
        ))
    })?;
    let dispatch = dispatch(capsule).map_err(PyPolarsErr::from)?;

    let cloud_options = storage_options
        .extract_opt_cloud_options(CloudScheme::from_path(source_url), credential_provider)?;
    let cast_columns_policy = cast_options.0;

    let dsl = match dispatch {
        Dispatch::V1(plugin) => {
            let request = iceberg_v1::Request {
                metadata_location,
                snapshot_id,
                from_snapshot_id_exclusive,
                to_snapshot_id_inclusive,
                projection,
                filter_columns,
                row_filter,
                limit,
                max_threads: Some(polars_config::config().max_threads() as u64),
                use_metadata_statistics,
                fast_deletion_count,
                verbose,
                testing_fail,
            };

            py.enter_polars(move || {
                v1::scan(
                    plugin,
                    PLUGIN_NAME,
                    &request,
                    cloud_options,
                    cast_columns_policy,
                )
            })?
        },
    };

    Ok(polars::prelude::LazyFrame::from(dsl).into())
}
