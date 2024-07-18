use polars::export::arrow::array::Array;
use polars::export::arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyType};

use super::*;

/// Validate PyCapsule has provided name
fn validate_pycapsule_name(capsule: &Bound<PyCapsule>, expected_name: &str) -> PyResult<()> {
    let capsule_name = capsule.name()?;
    if let Some(capsule_name) = capsule_name {
        let capsule_name = capsule_name.to_str()?;
        if capsule_name != expected_name {
            return Err(PyValueError::new_err(format!(
                "Expected name '{}' in PyCapsule, instead got '{}'",
                expected_name, capsule_name
            )));
        }
    } else {
        return Err(PyValueError::new_err(
            "Expected schema PyCapsule to have name set.",
        ));
    }

    Ok(())
}

/// Import `__arrow_c_stream__` across Python boundary.
fn call_arrow_c_stream<'py>(ob: &'py Bound<PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    if !ob.hasattr("__arrow_c_stream__")? {
        return Err(PyValueError::new_err(
            "Expected an object with dunder __arrow_c_stream__",
        ));
    }

    let capsule = ob.getattr("__arrow_c_stream__")?.call0()?.downcast_into()?;
    Ok(capsule)
}

pub(crate) fn import_stream_pycapsule(capsule: &Bound<PyCapsule>) -> PyResult<PySeries> {
    validate_pycapsule_name(capsule, "arrow_array_stream")?;

    // Takes ownership of the pointed to ArrowArrayStream
    // This acts to move the data out of the capsule pointer, setting the release callback to NULL
    let stream_ptr =
        Box::new(unsafe { std::ptr::replace(capsule.pointer() as _, ArrowArrayStream::empty()) });

    let mut stream = unsafe {
        ArrowArrayStreamReader::try_new(stream_ptr)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    };

    let mut produced_arrays: Vec<Box<dyn Array>> = vec![];
    while let Some(array) = unsafe { stream.next() } {
        produced_arrays.push(array.unwrap());
    }

    let s = Series::try_from((stream.field(), produced_arrays)).unwrap();
    Ok(PySeries::new(s))
}
#[pymethods]
impl PySeries {
    #[classmethod]
    pub fn from_arrow_c_stream(_cls: &Bound<PyType>, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capsule = call_arrow_c_stream(ob)?;
        import_stream_pycapsule(&capsule)
    }
}
