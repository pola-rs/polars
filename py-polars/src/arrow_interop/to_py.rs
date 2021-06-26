use libc::uintptr_t;
use polars_core::utils::arrow::{array::ArrayRef, ffi, record_batch::RecordBatch};
use pyo3::prelude::*;

/// Arrow array to Python.
pub fn to_py_array(array: &ArrayRef, py: Python, pyarrow: &PyModule) -> PyResult<PyObject> {
    let ffi_array = ffi::export_to_c(array.clone()).expect("c ptr");
    let (array_ptr, schema_ptr) = ffi_array.references();
    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as uintptr_t, schema_ptr as uintptr_t),
    )?;
    Ok(array.to_object(py))
}

/// RecordBatch to Python.
pub fn to_py_rb(rb: &RecordBatch, py: Python, pyarrow: &PyModule) -> PyResult<PyObject> {
    let mut arrays = Vec::with_capacity(rb.num_columns());
    let mut names = Vec::with_capacity(rb.num_columns());

    let schema = rb.schema();
    for (array, field) in rb.columns().iter().zip(schema.fields()) {
        let array_object = to_py_array(array, py, pyarrow)?;
        arrays.push(array_object);
        names.push(field.name());
    }

    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (arrays, names))?;

    Ok(record.to_object(py))
}
