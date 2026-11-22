use polars::datatypes::CompatLevel;
use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField, PlSmallStr, SchemaExt};
use polars::series::Series;
use polars_arrow::array::StructArray;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::ffi;
use polars_arrow::record_batch::RecordBatch;
use polars_core::utils::polars_arrow;
use polars_error::PolarsResult;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

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

/// RecordBatch to Python.
pub(crate) fn to_py_rb(
    rb: &RecordBatch,
    py: Python<'_>,
    pyarrow: &Bound<PyModule>,
) -> PyResult<Py<PyAny>> {
    let mut arrays = Vec::with_capacity(rb.width());

    for (array, field) in rb.columns().iter().zip(rb.schema().iter_values()) {
        let array_object = to_py_array(array.clone(), field, pyarrow)?;
        arrays.push(array_object);
    }

    let schema = Box::new(ffi::export_field_to_c(&ArrowField {
        name: PlSmallStr::EMPTY,
        dtype: ArrowDataType::Struct(rb.schema().iter_values().cloned().collect()),
        is_nullable: false,
        metadata: None,
    }));
    let schema_ptr: *const ffi::ArrowSchema = &*schema;

    let schema = pyarrow
        .getattr("Schema")?
        .call_method1("_import_from_c", (schema_ptr as Py_uintptr_t,))?;
    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (arrays, py.None(), schema))?;

    Ok(record.unbind())
}

/// Export a series to a C stream via a PyCapsule according to the Arrow PyCapsule Interface
/// https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
pub(crate) fn series_to_stream<'py>(
    series: &Series,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let field = series.field().to_arrow(CompatLevel::newest());
    let series = series.clone();
    let iter = Box::new(
        (0..series.n_chunks())
            .flat_map(move |i| series.try_to_arrow_split(i, CompatLevel::newest())),
    ) as _;

    let stream = ffi::export_iterator(iter, field);
    PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
}

pub(crate) fn dataframe_to_stream<'py>(
    df: &DataFrame,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let schema = df.schema().to_arrow(CompatLevel::newest());
    let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());
    let field = ArrowField::new(PlSmallStr::EMPTY, dtype.clone(), false);
    let stream = ffi::export_iterator(frame_to_struct_arrays(df.clone(), dtype, false), field);
    PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
}

#[cfg(feature = "c_api")]
#[pyfunction]
pub(crate) fn polars_schema_to_pycapsule<'py>(
    py: Python<'py>,
    schema: crate::prelude::Wrap<polars::prelude::Schema>,
    compat_level: crate::prelude::PyCompatLevel,
) -> PyResult<Bound<'py, PyCapsule>> {
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

/// The record batches of `df` as arrow struct arrays of `dtype`, the way a C stream carries a table.
pub(crate) fn frame_to_struct_arrays(
    df: DataFrame,
    dtype: ArrowDataType,
    parallel: bool,
) -> Box<dyn Iterator<Item = PolarsResult<ArrayRef>> + Send> {
    // `into_try_iter_chunks` cuts a frame without columns into batches of the ideal morsel size.
    if df.width() == 0 {
        let height = df.height();
        let array = (height > 0).then(|| Ok(StructArray::new(dtype, height, vec![], None).boxed()));
        return Box::new(array.into_iter());
    }

    let batches = df.into_try_iter_chunks(CompatLevel::newest(), parallel);
    Box::new(batches.map(move |batch| {
        let batch = batch?;
        let array = StructArray::new(dtype.clone(), batch.len(), batch.into_arrays(), None);
        Ok(array.boxed())
    }))
}
