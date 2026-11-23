mod convert;
mod host;
mod schema;
use std::ffi::CStr;
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
use std::sync::Arc;

use polars::prelude::CloudScheme;
use polars::prelude::deletion::{DeltaDeletionVectorDecoder, set_delta_deletion_vector_decoder};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::SharedStorage;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err};
use polars_io_ext_ffi::common::{FfiError, FfiErrorKind, FfiSlice};
use polars_io_ext_ffi::delta_v1;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use super::iceberg_plugin::metadata_cache::{PyIcebergMetadataFileCache, ScopedMetadataCache};
use crate::error::PyPolarsErr;
use crate::io::cloud_options::OptPyCloudOptions;
use crate::lazyframe::PyLazyFrame;
use crate::utils::EnterPolarsExt;

/// Delta plugin IDs this Polars can call, oldest first. Removing an ID drops support for
/// plugins that only provide it.
pub const SUPPORTED_IDS: &[&CStr] = &[delta_v1::ID];

/// Python distribution name of the plugin, for error messages.
const PLUGIN_NAME: &str = "polars_delta";

enum Dispatch {
    V2(&'static delta_v1::Plugin),
}

fn dispatch(capsule: &Bound<'_, PyCapsule>) -> PolarsResult<Dispatch> {
    // SAFETY: Plugin capsule names are static; the name is only used during this call.
    let name = capsule
        .name()
        .map_err(|e| polars_err!(ComputeError: "invalid {PLUGIN_NAME} capsule: {e}"))?
        .map(|n| unsafe { n.as_cstr() }.to_owned());

    match name.as_deref() {
        Some(id) if id == delta_v1::ID => {
            let ptr = capsule
                .pointer_checked(Some(id))
                .map_err(|e| polars_err!(ComputeError: "invalid {PLUGIN_NAME} capsule: {e}"))?;
            // SAFETY: A capsule with this name points to a static `delta_v1::Plugin`, and plugin
            // extension modules are never unloaded.
            Ok(Dispatch::V2(unsafe {
                &*(ptr.as_ptr() as *const delta_v1::Plugin)
            }))
        },
        _ => polars_bail!(
            ComputeError:
            "unsupported {PLUGIN_NAME} plugin ID {}; this Polars supports {:?}",
            name.map_or_else(|| "<none>".to_owned(), |n| format!("{:?}", n.to_string_lossy())),
            SUPPORTED_IDS.iter().map(|id| id.to_str().unwrap()).collect::<Vec<_>>()
        ),
    }
}

/// Serves deletion-vector decoding from the plugin's vtable; registered with the host's scan
/// engine, which does all I/O itself.
struct PluginDecoder(&'static delta_v1::Plugin);

fn keep_mask_to_bitmap(mask: delta_v1::KeepMask) -> PolarsResult<Bitmap> {
    let len = usize::try_from(mask.len)
        .ok()
        .filter(|len| len.div_ceil(8) <= mask.bytes.len())
        .ok_or_else(|| polars_err!(ComputeError: "{PLUGIN_NAME}: invalid keep mask length"))?;
    // The buffer stays owned by the plugin's allocator (released through its callback when the
    // bitmap's last reference drops): no copy. The plugin module is never unloaded.
    let storage = SharedStorage::from_owner(mask.bytes);
    // SAFETY: `len` bits fit in the storage, checked above.
    Ok(unsafe { Bitmap::from_inner_unchecked(storage, 0, len, None) })
}

fn decode_err(e: FfiError) -> PolarsError {
    polars_err!(ComputeError: "{}", e.message())
}

impl DeltaDeletionVectorDecoder for PluginDecoder {
    fn decode_inline(
        &self,
        data: &[u8],
        size: u32,
        cardinality: u64,
        max_rows: usize,
    ) -> PolarsResult<Bitmap> {
        let mask = unsafe {
            (self.0.decode_inline)(FfiSlice::new(data), size, cardinality, max_rows as u64)
        }
        .into_result()
        .map_err(decode_err)?;
        keep_mask_to_bitmap(mask)
    }

    fn decode_frame(
        &self,
        frame: &[u8],
        offset: u64,
        expected_size: u32,
        cardinality: u64,
        max_rows: usize,
    ) -> PolarsResult<Bitmap> {
        let mask = unsafe {
            (self.0.decode_frame)(
                FfiSlice::new(frame),
                offset,
                expected_size,
                cardinality,
                max_rows as u64,
            )
        }
        .into_result()
        .map_err(decode_err)?;
        keep_mask_to_bitmap(mask)
    }

    fn check_file_version(&self, header: &[u8]) -> PolarsResult<()> {
        unsafe { (self.0.check_file_version)(FfiSlice::new(header)) }
            .into_result()
            .map_err(decode_err)
    }
}

fn register_decoder(plugin: &'static delta_v1::Plugin) {
    set_delta_deletion_vector_decoder(Arc::new(PluginDecoder(plugin)));
}

/// Make scans find the decoder in a process that did not resolve the plan: on first need, import
/// the Python layer and let it load the installed plugin. Called at module init.
pub fn install_decoder_loader() {
    polars::prelude::deletion::set_delta_deletion_vector_decoder_loader(Box::new(|| {
        // This runs on a blocking-pool thread that can outlive the interpreter (a scan collected
        // from a daemon thread or an atexit hook): `attach` would hang or crash there.
        // `attach` is reentrant for a thread that already holds the GIL.
        Python::try_attach(|py| {
            let reason = py
                .import("polars.io.delta._plugin")
                .and_then(|m| m.call_method0("dv_decoder_load_error"))
                .and_then(|r| r.extract::<Option<String>>());
            match reason {
                Ok(None) => Ok(()),
                Ok(Some(why)) => Err(why),
                Err(e) => Err(format!("failed to load: {e}")),
            }
        })
        .unwrap_or_else(|| Err("the Python interpreter is shutting down".to_string()))
    }));
}

/// Register the plugin's deletion-vector decoder with the scan engine. Scans resolved by
/// `deltalake` can have native descriptors too, so the Python layer calls this whenever the
/// plugin is installed.
#[pyfunction]
pub fn _delta_plugin_register_decoder(capsule: &Bound<'_, PyAny>) -> PyResult<()> {
    let capsule = capsule.cast::<PyCapsule>()?;
    let Dispatch::V2(plugin) = dispatch(capsule).map_err(PyPolarsErr::from)?;
    register_decoder(plugin);
    Ok(())
}

/// Whether a deletion-vector decoder is registered.
#[pyfunction]
pub fn _delta_plugin_decoder_registered() -> bool {
    polars::prelude::deletion::delta_deletion_vector_decoder().is_some()
}

#[pyfunction]
#[pyo3(signature = (capsule, *, table_uri, version, timestamp_ms, projection, filter_columns, statistics_columns, limit, use_metadata_statistics, schema_only=false, verbose, testing_fail, storage_options, credential_provider, metadata_cache))]
#[allow(clippy::too_many_arguments)]
pub fn _delta_plugin_scan(
    py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
    table_uri: String,
    version: Option<i64>,
    timestamp_ms: Option<i64>,
    projection: Option<Vec<String>>,
    filter_columns: Option<Vec<String>>,
    statistics_columns: Option<Vec<String>>,
    limit: Option<u64>,
    use_metadata_statistics: bool,
    schema_only: bool,
    verbose: bool,
    testing_fail: Option<String>,
    storage_options: OptPyCloudOptions,
    credential_provider: Option<Py<PyAny>>,
    metadata_cache: PyRef<'_, PyIcebergMetadataFileCache>,
) -> PyResult<(PyLazyFrame, i64)> {
    let capsule = capsule.cast::<PyCapsule>()?;
    let Dispatch::V2(plugin) = dispatch(capsule).map_err(PyPolarsErr::from)?;
    register_decoder(plugin);
    let cloud_options = storage_options
        .extract_opt_cloud_options(CloudScheme::from_path(&table_uri), credential_provider)?;
    let request = delta_v1::Request {
        table_uri,
        version,
        timestamp_ms,
        projection,
        filter_columns,
        statistics_columns,
        limit,
        max_threads: Some(polars_config::config().max_threads() as u64),
        use_metadata_statistics,
        fast_deletion_count: true,
        schema_only,
        verbose,
        testing_fail,
    };
    let metadata_cache = ScopedMetadataCache {
        cache: metadata_cache.0.clone(),
        scope: BuildHasherDefault::<DefaultHasher>::default()
            .hash_one(&cloud_options)
            .to_string(),
        stats: Default::default(),
    };
    let (lf, version) = py.enter_polars(move || {
        let ctx = host::HostCtx {
            cloud_options: cloud_options.clone(),
            metadata_cache: Some(metadata_cache),
            io_error_kinds: Default::default(),
        };
        let host = ctx.host();
        let output = request
            .with_ffi(|r| unsafe { (plugin.plan)(&host, r) })
            .into_result()
            .map_err(|e| {
                let io_kind = ctx.io_error_kinds.kind_of(&e.message());
                ffi_to_polars_err(e, PLUGIN_NAME, io_kind)
            })?;
        if schema_only {
            // An empty frame with the table's schema: the plan listed no files.
            let (schema, version) = schema::schema_only(output)?;
            let lf = polars::prelude::IntoLazy::lazy(
                polars::prelude::DataFrame::empty_with_schema(&schema),
            );
            return PolarsResult::Ok((lf, version));
        }
        let (dsl, version) = convert::build_scan(output, cloud_options)?;
        Ok((polars::prelude::LazyFrame::from(dsl), version))
    })?;
    Ok((lf.into(), version))
}

fn ffi_to_polars_err(
    e: FfiError,
    plugin: &str,
    io_kind: Option<std::io::ErrorKind>,
) -> PolarsError {
    if e.kind() == FfiErrorKind::INVALID_INPUT {
        // Invalid user input (e.g. an unknown snapshot ID) is reported as a `ValueError` with the
        // plugin's message.
        return pyo3::exceptions::PyValueError::new_err(e.message()).into();
    }
    if e.kind() == FfiErrorKind::NOT_IMPLEMENTED {
        // Unsupported table features are raised as `NotImplementedError`,
        // on which the Delta resolver may fall back unless the plugin is required.
        return pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "{plugin}: {}",
            e.message()
        ))
        .into();
    }
    let io_kind = match e.kind() {
        FfiErrorKind::NOT_FOUND => Some(std::io::ErrorKind::NotFound),
        FfiErrorKind::OBJECT_CHANGED => Some(std::io::ErrorKind::Other),
        // Storage errors keep their Python exception type (e.g. `PermissionError`), as with the
        // deltalake planner.
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
