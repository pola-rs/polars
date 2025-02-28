use arrow::datatypes::IntegerType;
use arrow::record_batch::RecordBatch;
use polars::prelude::*;
use polars_compute::cast::CastOptionsImpl;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use super::PyDataFrame;
use crate::conversion::{ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::interop;
use crate::interop::arrow::to_py::dataframe_to_stream;
use crate::prelude::PyCompatLevel;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    #[cfg(feature = "object")]
    pub fn row_tuple<'py>(&self, idx: i64, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= self.df.height() {
            return Err(PyPolarsErr::from(polars_err!(oob = idx, self.df.height())).into());
        }
        PyTuple::new(
            py,
            self.df.get_columns().iter().map(|s| match s.dtype() {
                DataType::Object(_, _) => {
                    let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                    obj.into_py_any(py).unwrap()
                },
                _ => Wrap(s.get(idx).unwrap()).into_py_any(py).unwrap(),
            }),
        )
    }

    #[cfg(feature = "object")]
    pub fn row_tuples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut rechunked;
        // Rechunk if random access would become rather expensive.
        // TODO: iterate over the chunks directly instead of using random access.
        let df = if self.df.max_n_chunks() > 16 {
            rechunked = self.df.clone();
            rechunked.as_single_chunk_par();
            &rechunked
        } else {
            &self.df
        };
        PyList::new(
            py,
            (0..df.height()).map(|idx| {
                PyTuple::new(
                    py,
                    df.get_columns().iter().map(|c| match c.dtype() {
                        DataType::Null => py.None(),
                        DataType::Object(_, _) => {
                            let obj: Option<&ObjectValue> = c.get_object(idx).map(|any| any.into());
                            obj.into_py_any(py).unwrap()
                        },
                        _ => {
                            // SAFETY: we are in bounds.
                            let av = unsafe { c.get_unchecked(idx) };
                            Wrap(av).into_py_any(py).unwrap()
                        },
                    }),
                )
                .unwrap()
            }),
        )
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_arrow(&mut self, py: Python, compat_level: PyCompatLevel) -> PyResult<Vec<PyObject>> {
        py.enter_polars_ok(|| self.df.align_chunks_par())?;
        let pyarrow = py.import("pyarrow")?;

        let rbs = self
            .df
            .iter_chunks(compat_level.0, true)
            .map(|rb| interop::arrow::to_py::to_py_rb(&rb, py, &pyarrow))
            .collect::<PyResult<_>>()?;
        Ok(rbs)
    }

    /// Create a `Vec` of PyArrow RecordBatch instances.
    ///
    /// Note this will give bad results for columns with dtype `pl.Object`,
    /// since those can't be converted correctly via PyArrow. The calling Python
    /// code should make sure these are not included.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_pandas(&mut self, py: Python) -> PyResult<Vec<PyObject>> {
        py.enter_polars_ok(|| self.df.as_single_chunk_par())?;
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;
            let cat_columns = self
                .df
                .get_columns()
                .iter()
                .enumerate()
                .filter(|(_i, s)| {
                    matches!(
                        s.dtype(),
                        DataType::Categorical(_, _) | DataType::Enum(_, _)
                    )
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            let enum_and_categorical_dtype = ArrowDataType::Dictionary(
                IntegerType::Int64,
                Box::new(ArrowDataType::LargeUtf8),
                false,
            );

            let mut replaced_schema = None;
            let rbs = self
                .df
                .iter_chunks(CompatLevel::oldest(), true)
                .map(|rb| {
                    let length = rb.len();
                    let (schema, mut arrays) = rb.into_schema_and_arrays();

                    // Pandas does not allow unsigned dictionary indices so we replace them.
                    replaced_schema =
                        (replaced_schema.is_none() && !cat_columns.is_empty()).then(|| {
                            let mut schema = schema.as_ref().clone();
                            for i in &cat_columns {
                                let (_, field) = schema.get_at_index_mut(*i).unwrap();
                                field.dtype = enum_and_categorical_dtype.clone();
                            }
                            Arc::new(schema)
                        });

                    for i in &cat_columns {
                        let arr = arrays.get_mut(*i).unwrap();
                        let out = polars_compute::cast::cast(
                            &**arr,
                            &enum_and_categorical_dtype,
                            CastOptionsImpl::default(),
                        )
                        .unwrap();
                        *arr = out;
                    }
                    let schema = replaced_schema
                        .as_ref()
                        .map_or(schema, |replaced| replaced.clone());
                    let rb = RecordBatch::new(length, schema, arrays);

                    interop::arrow::to_py::to_py_rb(&rb, py, &pyarrow)
                })
                .collect::<PyResult<_>>()?;
            Ok(rbs)
        })
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &'py mut self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        py.enter_polars_ok(|| self.df.align_chunks_par())?;
        dataframe_to_stream(&self.df, py)
    }
}
