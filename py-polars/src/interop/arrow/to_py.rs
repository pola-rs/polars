use std::ffi::CString;

use arrow::datatypes::ArrowDataType;
use arrow::ffi;
use arrow::record_batch::RecordBatch;
use polars::datatypes::CompatLevel;
use polars::frame::DataFrame;
use polars::prelude::{ArrayRef, ArrowField};
use polars::series::Series;
use polars_core::utils::arrow;
use polars_error::PolarsResult;
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
    let field = series.field().to_arrow(CompatLevel::newest());
    let iter = Box::new(series.chunks().clone().into_iter().map(Ok)) as _;
    let stream = ffi::export_iterator(iter, field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new_bound(py, stream, Some(stream_capsule_name))
}

pub(crate) fn dataframe_to_stream<'py>(
    df: &'py DataFrame,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let iter = Box::new(DataFrameStreamIterator::new(df));
    let field = iter.field();
    let stream = ffi::export_iterator(iter, field);
    let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
    PyCapsule::new_bound(py, stream, Some(stream_capsule_name))
}

pub struct DataFrameStreamIterator {
    columns: Vec<Series>,
    data_type: ArrowDataType,
    idx: usize,
    n_chunks: usize,
}

impl DataFrameStreamIterator {
    fn new(df: &DataFrame) -> Self {
        let schema = df.schema().to_arrow(CompatLevel::newest());
        let data_type = ArrowDataType::Struct(schema.fields);

        Self {
            columns: df.get_columns().to_vec(),
            data_type,
            idx: 0,
            n_chunks: df.n_chunks(),
        }
    }

    fn field(&self) -> ArrowField {
        ArrowField::new("", self.data_type.clone(), false)
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
                .collect();
            self.idx += 1;

            let array = arrow::array::StructArray::new(self.data_type.clone(), batch_cols, None);
            Some(Ok(Box::new(array)))
        }
    }
}
