use arrow::ffi;
use polars::prelude::{ArrayRef, ArrowField};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;

/// Arrow array to Python.
pub(crate) fn to_py_array<'py>(
    array: ArrayRef,
    pyarrow: Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    let schema = Box::new(ffi::export_field_to_c(&ArrowField::new(
        "".into(),
        array.dtype().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    pyarrow.getattr("Array")?.call_method1(
        "_import_arrow_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )
}
