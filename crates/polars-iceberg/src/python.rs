//! Python extension module (`polars_iceberg` when built with feature `iceberg`). It only exposes the plugin contracts as PyCapsules;
//! Polars calls them directly without going through Python.
//!
//! The handshake (`_polars_io_plugin_ids`, `_capsule(id)`) is shared by all Polars I/O plugins
//! and never changes (see `polars_io_ext_ffi`).
// Without a format feature `PluginId` is uninhabited and the capsule code is dead.
#![cfg_attr(not(feature = "iceberg"), allow(unused, unreachable_code))]
use std::ffi::CStr;
use std::ptr::NonNull;

use polars_io_ext_ffi::PluginId;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

/// Plugin IDs exported by this build, oldest first.
const PLUGIN_IDS: &[PluginId] = PluginId::ALL;

/// Returns the PyCapsule of the contract with the given plugin ID.
#[pyfunction]
fn _capsule<'py>(py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyCapsule>> {
    let Some(plugin_id) = PluginId::from_id_str(id).filter(|id| PLUGIN_IDS.contains(id)) else {
        return Err(PyValueError::new_err(format!(
            "{} does not provide plugin ID {id:?}",
            crate::PACKAGE_NAME
        )));
    };
    let plugin: NonNull<std::ffi::c_void> = match plugin_id {
        #[cfg(feature = "iceberg")]
        PluginId::IcebergV1 => NonNull::from(&crate::iceberg::PLUGIN_V1).cast(),
    };
    // SAFETY: The struct is a static; the extension module is never unloaded.
    unsafe { PyCapsule::new_with_pointer(py, plugin, plugin_id.name()) }
}

/// Returns a capsule with the `iceberg.v1` contract under a different name, for testing the
/// refusal of unknown IDs by Polars.
#[cfg(feature = "iceberg")]
#[pyfunction]
fn _capsule_for_testing<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyCapsule>> {
    let name: &'static CStr = Box::leak(
        std::ffi::CString::new(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_boxed_c_str(),
    );
    unsafe {
        PyCapsule::new_with_pointer(py, NonNull::from(&crate::iceberg::PLUGIN_V1).cast(), name)
    }
}

/// The module name must match `module-name` of the wheel (see `pyproject.toml`).
#[pymodule]
#[cfg_attr(feature = "iceberg", pyo3(name = "polars_iceberg"))]
fn polars_io_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "_polars_io_plugin_ids",
        PyTuple::new(m.py(), PLUGIN_IDS.iter().map(|id| id.as_str()))?,
    )?;
    m.add_function(wrap_pyfunction!(_capsule, m)?)?;
    #[cfg(feature = "iceberg")]
    m.add_function(wrap_pyfunction!(_capsule_for_testing, m)?)?;
    Ok(())
}
