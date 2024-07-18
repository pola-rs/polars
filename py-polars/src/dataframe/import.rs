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

pub(crate) fn import_stream_pycapsule(capsule: &Bound<PyCapsule>) -> PyResult<PyDataFrame> {
    validate_pycapsule_name(capsule, "arrow_array_stream")?;

    // Takes ownership of the pointed to ArrowArrayStream
    // This acts to move the data out of the capsule pointer, setting the release callback to NULL
    let stream_ptr =
        Box::new(unsafe { std::ptr::replace(capsule.pointer() as _, ArrowArrayStream::empty()) });

    let mut stream = unsafe {
        ArrowArrayStreamReader::try_new(stream_ptr)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    };

    // For now we'll assume that these are struct arrays to represent record batches
    let mut produced_arrays = vec![];
    while let Some(array) = unsafe { stream.next() } {
        let arr = array.map_err(|err| PyValueError::new_err(err.to_string()))?;
        let struct_arr = match arr.data_type() {
            ArrowDataType::Struct(_) => arr.as_any().downcast_ref::<StructArray>().unwrap().clone(),
            _ => return Err(PyValueError::new_err("Expected struct data type")),
        };
        produced_arrays.push(struct_arr);
    }

    let stream_field = stream.field();
    // For now we'll assume that these are struct arrays to represent record batches
    let struct_fields = match stream_field.data_type() {
        ArrowDataType::Struct(struct_fields) => struct_fields,
        _ => return Err(PyValueError::new_err("Expected struct data type")),
    };

    let mut columns: Vec<Series> = vec![];
    for (col_idx, column_field) in struct_fields.iter().enumerate() {
        let column_chunks = produced_arrays
            .iter()
            .map(|arr| arr.values()[col_idx].clone())
            .collect::<Vec<_>>();
        // TODO: remove unwrap
        columns.push(Series::try_from((column_field, column_chunks)).unwrap());
    }

    // TODO: remove unwrap
    Ok(PyDataFrame::new(
        polars::frame::DataFrame::new(columns).unwrap(),
    ))
}
#[pymethods]
impl PyDataFrame {
    #[classmethod]
    pub fn from_arrow_c_stream(_cls: &Bound<PyType>, ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capsule = call_arrow_c_stream(ob)?;
        import_stream_pycapsule(&capsule)
    }
}
