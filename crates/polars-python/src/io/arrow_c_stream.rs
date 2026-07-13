use arrow::array::{Array, StructArray};
use arrow::ffi::{ArrowArrayStream, ArrowArrayStreamReader};
use parking_lot::Mutex;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::series::{call_arrow_c_stream, open_stream_capsule};

struct ReaderState {
    reader: ArrowArrayStreamReader<Box<ArrowArrayStream>>,
    // Resolved from the first `next_batch` call's `with_columns` and reused
    // after: the engine calls `next_batch` with the same projection on every
    // batch of a given scan, so re-parsing it per batch would be wasted work.
    projection: Option<Option<PlHashSet<PlSmallStr>>>,
}

#[pyclass]
pub struct PyArrowCStreamReader {
    state: Mutex<ReaderState>,
    schema: Schema,
}

#[pymethods]
impl PyArrowCStreamReader {
    #[new]
    fn new(ob: &Bound<PyAny>) -> PyResult<Self> {
        let capsule = call_arrow_c_stream(ob)?;
        let reader = open_stream_capsule(&capsule)?;

        let ArrowDataType::Struct(fields) = &reader.field().dtype else {
            return Err(PyValueError::new_err(
                "Arrow C Stream schema must be a struct type",
            ));
        };
        let schema = Schema::from_iter(fields.iter().map(Field::from));

        Ok(Self {
            state: Mutex::new(ReaderState {
                reader,
                projection: None,
            }),
            schema,
        })
    }

    #[getter]
    fn schema(&self) -> Wrap<Schema> {
        Wrap(self.schema.clone())
    }

    fn next_batch(&self, with_columns: Option<Vec<PlSmallStr>>) -> PyResult<Option<PyDataFrame>> {
        let mut state = self.state.lock();
        if state.projection.is_none() {
            state.projection = Some(with_columns.map(|cols| cols.into_iter().collect()));
        }

        let array = match unsafe { state.reader.next() } {
            Some(Ok(array)) => array,
            Some(Err(e)) => return Err(PyPolarsErr::from(e).into()),
            None => return Ok(None),
        };

        let projection = state.projection.as_ref().unwrap().as_ref();
        let df = struct_array_to_df(array, projection).map_err(PyPolarsErr::from)?;
        Ok(Some(PyDataFrame::new(df)))
    }
}

fn struct_array_to_df(
    array: Box<dyn Array>,
    projection: Option<&PlHashSet<PlSmallStr>>,
) -> PolarsResult<DataFrame> {
    let struct_array = array.as_any().downcast_ref::<StructArray>().ok_or_else(
        || polars_err!(ComputeError: "expected a StructArray from the Arrow C Stream"),
    )?;

    let columns = struct_array
        .values()
        .iter()
        .zip(struct_array.fields())
        .filter(|(_, field)| projection.is_none_or(|proj| proj.contains(&field.name)))
        .map(|(arr, field)| unsafe {
            Series::_try_from_arrow_unchecked(field.name.clone(), vec![arr.clone()], arr.dtype())
                .map(Series::into_column)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    DataFrame::new_infer_height(columns)
}
