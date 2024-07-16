use std::ffi::CString;

use arrow::ffi;
use arrow::record_batch::RecordBatch;
use polars::datatypes::{CompatLevel, DataType, Field};
use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField};
use polars::series::Series;
use polars_core::utils::arrow;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Arrow array to Python.
pub(crate) fn to_py_array(
    array: ArrayRef,
    py: Python,
    pyarrow: &Bound<PyModule>,
) -> PyResult<PyObject> {
    let schema = Box::new(ffi::export_field_to_c(&ArrowField::new(
        "",
        array.data_type().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.to_object(py))
}

/// RecordBatch to Python.
pub(crate) fn to_py_rb(
    rb: &RecordBatch,
    names: &[&str],
    py: Python,
    pyarrow: &Bound<PyModule>,
) -> PyResult<PyObject> {
    let mut arrays = Vec::with_capacity(rb.len());

    for array in rb.columns() {
        let array_object = to_py_array(array.clone(), py, pyarrow)?;
        arrays.push(array_object);
    }

    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (arrays, names.to_vec()))?;

    Ok(record.to_object(py))
}

/// Export a series to a C stream via a PyCapsule according to the Arrow PyCapsule Interface
/// https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
pub(crate) fn series_to_stream<'py>(
    series: &'py Series,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let field = series.field().to_arrow(CompatLevel::oldest());
    let iter = Box::new(series.chunks().clone().into_iter().map(Ok)) as _;
    let stream = ffi::export_iterator(iter, field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new_bound(py, stream, Some(stream_capsule_name))
}

pub(crate) fn dataframe_to_stream<'py>(
    df: &'py DataFrame,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let schema_fields = df.schema().iter_fields().collect::<Vec<_>>();

    let struct_field =
        Field::new("", DataType::Struct(schema_fields)).to_arrow(CompatLevel::oldest());
    let struct_data_type = struct_field.data_type().clone();

    let iter = df
        .iter_chunks(CompatLevel::oldest(), false)
        .into_iter()
        .map(|chunk| {
            let arrays = chunk.into_arrays();
            let x = arrow::array::StructArray::new(struct_data_type.clone(), arrays, None);
            Ok(Box::new(x) as Box<dyn arrow::array::Array>)
        });
    let stream = ffi::export_iterator(Box::new(iter), struct_field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new_bound(py, stream, Some(stream_capsule_name))
}
