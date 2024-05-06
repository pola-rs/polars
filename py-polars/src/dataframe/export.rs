use polars::export::arrow::record_batch::RecordBatch;
use polars_core::export::arrow::datatypes::IntegerType;
use polars_core::utils::arrow::compute::cast::CastOptions;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::*;
use crate::arrow_interop;
use crate::conversion::{ObjectValue, Wrap};

#[pymethods]
impl PyDataFrame {
    #[cfg(feature = "object")]
    pub fn row_tuple(&self, idx: i64) -> PyResult<PyObject> {
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= self.df.height() {
            return Err(PyPolarsErr::from(polars_err!(oob = idx, self.df.height())).into());
        }
        let out = Python::with_gil(|py| {
            PyTuple::new_bound(
                py,
                self.df.get_columns().iter().map(|s| match s.dtype() {
                    DataType::Object(_, _) => {
                        let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                        obj.to_object(py)
                    },
                    _ => Wrap(s.get(idx).unwrap()).into_py(py),
                }),
            )
            .into_py(py)
        });
        Ok(out)
    }

    #[cfg(feature = "object")]
    pub fn row_tuples(&self) -> PyObject {
        Python::with_gil(|py| {
            let df = &self.df;
            PyList::new_bound(
                py,
                (0..df.height()).map(|idx| {
                    PyTuple::new_bound(
                        py,
                        self.df.get_columns().iter().map(|s| match s.dtype() {
                            DataType::Null => py.None(),
                            DataType::Object(_, _) => {
                                let obj: Option<&ObjectValue> =
                                    s.get_object(idx).map(|any| any.into());
                                obj.to_object(py)
                            },
                            // SAFETY: we are in bounds.
                            _ => unsafe { Wrap(s.get_unchecked(idx)).into_py(py) },
                        }),
                    )
                }),
            )
            .into_py(py)
        })
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_arrow(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.align_chunks();
        Python::with_gil(|py| {
            let pyarrow = py.import_bound("pyarrow")?;
            let names = self.df.get_column_names();

            let rbs = self
                .df
                .iter_chunks(false)
                .map(|rb| arrow_interop::to_py::to_py_rb(&rb, &names, py, &pyarrow))
                .collect::<PyResult<_>>()?;
            Ok(rbs)
        })
    }

    /// Create a `Vec` of PyArrow RecordBatch instances.
    ///
    /// Note this will give bad results for columns with dtype `pl.Object`,
    /// since those can't be converted correctly via PyArrow. The calling Python
    /// code should make sure these are not included.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_pandas(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.as_single_chunk_par();
        Python::with_gil(|py| {
            let pyarrow = py.import_bound("pyarrow")?;
            let names = self.df.get_column_names();
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
            let rbs = self
                .df
                .iter_chunks(false)
                .map(|rb| {
                    let mut rb = rb.into_arrays();
                    for i in &cat_columns {
                        let arr = rb.get_mut(*i).unwrap();
                        let out = polars_core::export::arrow::compute::cast::cast(
                            &**arr,
                            &ArrowDataType::Dictionary(
                                IntegerType::Int64,
                                Box::new(ArrowDataType::LargeUtf8),
                                false,
                            ),
                            CastOptions::default(),
                        )
                        .unwrap();
                        *arr = out;
                    }
                    let rb = RecordBatch::new(rb);

                    arrow_interop::to_py::to_py_rb(&rb, &names, py, &pyarrow)
                })
                .collect::<PyResult<_>>()?;
            Ok(rbs)
        })
    }
}
