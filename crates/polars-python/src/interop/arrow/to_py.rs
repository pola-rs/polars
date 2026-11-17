use std::sync::{Arc, Mutex};

use polars::datatypes::CompatLevel;
use polars::prelude::{ArrayRef, ArrowField, ArrowSchema, PlSmallStr};
use polars::series::Series;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi;
use polars_arrow::record_batch::RecordBatch;
use polars_core::frame::batches_to_struct_arrays;
use polars_core::utils::polars_arrow;
use polars_error::{PolarsResult, polars_err};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::error::PyPolarsErr;

/// Arrow array to Python.
pub(crate) fn to_py_array(
    array: ArrayRef,
    field: &ArrowField,
    pyarrow: &Bound<PyModule>,
) -> PyResult<Py<PyAny>> {
    let schema = Box::new(ffi::export_field_to_c(field));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.unbind())
}

/// Read `batches`, whose columns have the fields of `schema`, into a `pyarrow.Table` through an
/// Arrow C stream.
///
/// An export error is raised as the Polars error itself rather than as pyarrow's rewrapping
/// of its message.
pub(crate) fn batches_to_py_table(
    py: Python<'_>,
    batches: impl Iterator<Item = PolarsResult<RecordBatch>> + 'static,
    schema: ArrowSchema,
) -> PyResult<Py<PyAny>> {
    let first_error = Arc::new(Mutex::new(None));
    let error_slot = first_error.clone();
    let batches = batches.map(move |batch| {
        batch.map_err(|err| {
            let msg = err.to_string();
            error_slot.lock().unwrap().get_or_insert(err);
            polars_err!(ComputeError: "{msg}")
        })
    });

    let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());
    let field = ArrowField::new(PlSmallStr::EMPTY, dtype.clone(), false);
    let arrays = batches_to_struct_arrays(batches, dtype);
    let mut stream = Box::new(ffi::export_iterator(Box::new(arrays), field));
    let stream_ptr: *mut ffi::ArrowArrayStream = &mut *stream;
    // `read_all` releases the GIL while it pulls the batches.
    let table = py
        .import("pyarrow")?
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c", (stream_ptr as Py_uintptr_t,))?
        .call_method0("read_all");

    match table {
        Ok(table) => Ok(table.unbind()),
        Err(err) => match first_error.lock().unwrap().take() {
            Some(err) => Err(PyPolarsErr::from(err).into()),
            None => Err(err),
        },
    }
}

/// Export a series to a C stream via a PyCapsule according to the Arrow PyCapsule Interface
/// https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
pub(crate) fn series_to_stream<'py>(
    series: &Series,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let field = series.field().to_arrow(CompatLevel::newest());
    let series = series.clone();
    let iter = Box::new((0..series.n_chunks()).flat_map(move |i| {
        match series.try_to_arrow_split(i, CompatLevel::newest()) {
            Ok(arrays) => arrays.into_iter().map(Ok).collect(),
            Err(err) => vec![Err(err)],
        }
    })) as _;

    let stream = ffi::export_iterator(iter, field);
    PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
}

#[cfg(feature = "c_api")]
#[pyfunction]
pub(crate) fn polars_schema_to_pycapsule<'py>(
    py: Python<'py>,
    schema: crate::prelude::Wrap<polars::prelude::Schema>,
    compat_level: crate::prelude::PyCompatLevel,
) -> PyResult<Bound<'py, PyCapsule>> {
    use polars::prelude::SchemaExt;

    let schema: polars_arrow::ffi::ArrowSchema =
        polars_arrow::ffi::export_field_to_c(&ArrowField::new(
            PlSmallStr::EMPTY,
            ArrowDataType::Struct(
                schema
                    .0
                    .iter_fields()
                    .map(|x| x.to_arrow(compat_level.0))
                    .collect(),
            ),
            false,
        ));

    PyCapsule::new_with_value(py, schema, c"arrow_schema")
}
