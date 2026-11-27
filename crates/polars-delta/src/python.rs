//! Python extension module exposing the shared Polars I/O plugin handshake.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

/// Returns the contract capsule for a supported plugin ID.
#[pyfunction]
fn _capsule<'py>(py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyCapsule>> {
    if id != polars_io_ext_ffi::delta_v1::ID.to_str().unwrap() {
        return Err(PyValueError::new_err(format!("unknown plugin ID {id:?}")));
    }
    unsafe {
        PyCapsule::new_with_pointer(
            py,
            std::ptr::NonNull::from(&crate::delta::PLUGIN_V1).cast(),
            polars_io_ext_ffi::delta_v1::ID,
        )
    }
}

/// This name must match `module-name` in `pyproject.toml`.
#[pymodule]
fn polars_delta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // The commit the release workflow built from, to compare with the sources Polars is
    // released from. Empty outside of a release build.
    m.add(
        "__build_commit__",
        option_env!("POLARS_DELTA_BUILD_COMMIT").unwrap_or(""),
    )?;
    m.add(
        "_polars_io_plugin_ids",
        PyTuple::new(m.py(), ["polars.io_plugin.delta.v1"])?,
    )?;
    m.add_function(wrap_pyfunction!(_capsule, m)?)?;
    Ok(())
}
