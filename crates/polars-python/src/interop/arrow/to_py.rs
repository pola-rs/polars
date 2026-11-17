use polars::datatypes::CompatLevel;
use polars::prelude::{ArrayRef, ArrowField, ArrowSchema, PlSmallStr};
use polars::series::Series;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi;
use polars_arrow::record_batch::RecordBatch;
use polars_core::frame::batches_to_struct_arrays;
use polars_core::utils::polars_arrow;
use polars_error::PolarsResult;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::utils::EnterPolarsExt;

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

/// Arrow data type of `field` to Python.
pub(crate) fn to_py_dtype(field: &ArrowField, pyarrow: &Bound<PyModule>) -> PyResult<Py<PyAny>> {
    let schema = Box::new(ffi::export_field_to_c(field));
    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let dtype = pyarrow
        .getattr("DataType")?
        .call_method1("_import_from_c", (schema_ptr as Py_uintptr_t,))?;
    Ok(dtype.unbind())
}

/// Read `batches`, whose columns have the fields of `schema`, into a `pyarrow.Table`.
///
/// The batches are converted in [`EnterPolarsExt::enter_polars`], so that panics, interrupts and
/// export errors surface as they do for any other Polars call, and then handed to pyarrow as one
/// Arrow C stream.
pub(crate) fn batches_to_py_table(
    py: Python<'_>,
    batches: impl Iterator<Item = PolarsResult<RecordBatch>> + Send,
    schema: ArrowSchema,
) -> PyResult<Py<PyAny>> {
    let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());
    let field = ArrowField::new(PlSmallStr::EMPTY, dtype.clone(), false);
    let arrays = py.enter_polars(|| {
        batches_to_struct_arrays(batches, dtype).collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut stream = Box::new(ffi::export_iterator(
        Box::new(arrays.into_iter().map(Ok)),
        field,
    ));
    let stream_ptr: *mut ffi::ArrowArrayStream = &mut *stream;
    let table = py
        .import("pyarrow")?
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c", (stream_ptr as Py_uintptr_t,))?
        .call_method0("read_all")?;
    Ok(table.unbind())
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
