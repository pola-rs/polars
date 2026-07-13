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

#[pyclass]
pub struct PyArrowCStreamReader {
    reader: Mutex<ArrowArrayStreamReader<Box<ArrowArrayStream>>>,
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
            reader: Mutex::new(reader),
            schema,
        })
    }

    #[getter]
    fn schema(&self) -> Wrap<Schema> {
        Wrap(self.schema.clone())
    }

    fn next_batch(&self, with_columns: Option<Vec<PlSmallStr>>) -> PyResult<Option<PyDataFrame>> {
        match unsafe { self.reader.lock().next() } {
            Some(Ok(array)) => {
                let df = struct_array_to_df(array, &self.schema, with_columns.as_deref())
                    .map_err(PyPolarsErr::from)?;
                Ok(Some(PyDataFrame::new(df)))
            },
            Some(Err(e)) => Err(PyPolarsErr::from(e).into()),
            None => Ok(None),
        }
    }
}

fn struct_array_to_df(
    array: Box<dyn Array>,
    schema: &Schema,
    with_columns: Option<&[PlSmallStr]>,
) -> PolarsResult<DataFrame> {
    let struct_array = array.as_any().downcast_ref::<StructArray>().ok_or_else(
        || polars_err!(ComputeError: "expected a StructArray from the Arrow C Stream"),
    )?;
    let projection: Option<PlHashSet<&PlSmallStr>> = with_columns.map(|cols| cols.iter().collect());

    let columns = struct_array
        .values()
        .iter()
        .zip(schema.iter_names())
        .filter(|(_, name)| projection.as_ref().is_none_or(|proj| proj.contains(name)))
        .map(|(arr, name)| unsafe {
            Series::_try_from_arrow_unchecked(name.clone(), vec![arr.clone()], arr.dtype())
                .map(Series::into_column)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    DataFrame::new_infer_height(columns)
}
