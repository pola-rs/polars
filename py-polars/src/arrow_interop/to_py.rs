use polars::prelude::ArrowField;
use polars_core::utils::arrow::{array::ArrayRef, ffi, record_batch::RecordBatch};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;

/// Arrow array to Python.
pub(crate) fn to_py_array(array: ArrayRef, py: Python, pyarrow: &PyModule) -> PyResult<PyObject> {
    let array_ptr = Box::new(ffi::Ffi_ArrowArray::empty());
    let schema_ptr = Box::new(ffi::Ffi_ArrowSchema::empty());

    let array_ptr = Box::into_raw(array_ptr);
    let schema_ptr = Box::into_raw(schema_ptr);

    unsafe {
        ffi::export_field_to_c(
            &ArrowField::new("", array.data_type().clone(), true),
            schema_ptr,
        );
        ffi::export_array_to_c(array, array_ptr);
    };

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        Box::from_raw(array_ptr);
        Box::from_raw(schema_ptr);
    };

    Ok(array.to_object(py))
}

/// RecordBatch to Python.
pub(crate) fn to_py_rb(rb: &RecordBatch, py: Python, pyarrow: &PyModule) -> PyResult<PyObject> {
    let mut arrays = Vec::with_capacity(rb.num_columns());
    let mut names = Vec::with_capacity(rb.num_columns());

    let schema = rb.schema();
    for (array, field) in rb.columns().iter().zip(schema.fields()) {
        let array_object = to_py_array(array.clone(), py, pyarrow)?;
        arrays.push(array_object);
        names.push(field.name());
    }

    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (arrays, names))?;

    Ok(record.to_object(py))
}
