use std::ffi::CString;

use arrow::datatypes::ArrowDataType;
use arrow::ffi;
use arrow::record_batch::RecordBatch;
use polars::datatypes::CompatLevel;
use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField, PlSmallStr, SchemaExt};
use polars::series::Series;
use polars_core::utils::arrow;
use polars_error::PolarsResult;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Arrow array to Python.
pub(crate) fn to_py_array(
    array: ArrayRef,
    field: &ArrowField,
    pyarrow: &Bound<PyModule>,
) -> PyResult<PyObject> {
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
) -> PyResult<PyObject> {
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
    let iter = Box::new(series.chunks().clone().into_iter().map(Ok)) as _;
    let stream = ffi::export_iterator(iter, field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new(py, stream, Some(stream_capsule_name))
}

pub(crate) fn dataframe_to_stream<'py>(
    df: &DataFrame,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let iter = Box::new(DataFrameStreamIterator::new(df));
    let field = iter.field();
    let stream = ffi::export_iterator(iter, field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new(py, stream, Some(stream_capsule_name))
}

pub struct DataFrameStreamIterator {
    columns: Vec<Series>,
    dtype: ArrowDataType,
    idx: usize,
    n_chunks: usize,
}

impl DataFrameStreamIterator {
    fn new(df: &DataFrame) -> Self {
        let schema = df.schema().to_arrow(CompatLevel::newest());
        let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());

        Self {
            columns: df
                .get_columns()
                .iter()
                .map(|v| v.as_materialized_series().clone())
                .collect(),
            dtype,
            idx: 0,
            n_chunks: df.first_col_n_chunks(),
        }
    }

    fn field(&self) -> ArrowField {
        ArrowField::new(PlSmallStr::EMPTY, self.dtype.clone(), false)
    }
}

impl Iterator for DataFrameStreamIterator {
    type Item = PolarsResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            None
        } else {
            // create a batch of the columns with the same chunk no.
            let batch_cols = self
                .columns
                .iter()
                .map(|s| s.to_arrow(self.idx, CompatLevel::newest()))
                .collect::<Vec<_>>();
            self.idx += 1;

            let array = arrow::array::StructArray::new(
                self.dtype.clone(),
                batch_cols[0].len(),
                batch_cols,
                None,
            );
            Some(Ok(Box::new(array)))
        }
    }
}
