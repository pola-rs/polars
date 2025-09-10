use arrow::datatypes::IntegerType;
use arrow::record_batch::RecordBatch;
use parking_lot::RwLockWriteGuard;
use polars::prelude::*;
use polars_compute::cast::CastOptionsImpl;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};

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
        let df = self.df.read();
        let idx = if idx < 0 {
            (df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= df.height() {
            return Err(PyPolarsErr::from(polars_err!(oob = idx, df.height())).into());
        }
        PyTuple::new(
            py,
            df.get_columns().iter().map(|s| match s.dtype() {
                DataType::Object(_) => {
                    let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                    obj.into_py_any(py).unwrap()
                },
                _ => Wrap(s.get(idx).unwrap()).into_py_any(py).unwrap(),
            }),
        )
    }

    #[cfg(feature = "object")]
    pub fn row_tuples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let df = self.df.read();
        let mut rechunked;
        // Rechunk if random access would become rather expensive.
        // TODO: iterate over the chunks directly instead of using random access.
        let df = if df.max_n_chunks() > 16 {
            rechunked = df.clone();
            rechunked.as_single_chunk_par();
            &rechunked
        } else {
            &df
        };
        PyList::new(
            py,
            (0..df.height()).map(|idx| {
                PyTuple::new(
                    py,
                    df.get_columns().iter().map(|c| match c.dtype() {
                        DataType::Null => py.None(),
                        DataType::Object(_) => {
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
    pub fn to_arrow(&self, py: Python<'_>, compat_level: PyCompatLevel) -> PyResult<Vec<PyObject>> {
        let mut df = self.df.write();
        let dfr = &mut *df; // Lock guard isn't Send, but mut ref is.
        py.enter_polars_ok(|| dfr.align_chunks_par())?;
        let df = RwLockWriteGuard::downgrade(df);

        let pyarrow = py.import("pyarrow")?;

        let mut chunks = df.iter_chunks(compat_level.0, true);
        let mut rbs = Vec::with_capacity(chunks.size_hint().0);
        // df.iter_chunks() iteration could internally try to acquire the GIL on another thread,
        // so we make sure to run chunks.next() within enter_polars().
        while let Some(rb) = py.enter_polars_ok(|| chunks.next())? {
            let rb = interop::arrow::to_py::to_py_rb(&rb, py, &pyarrow)?;
            rbs.push(rb);
        }
        Ok(rbs)
    }

    /// Create a `Vec` of PyArrow RecordBatch instances.
    ///
    /// Note this will give bad results for columns with dtype `pl.Object`,
    /// since those can't be converted correctly via PyArrow. The calling Python
    /// code should make sure these are not included.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_pandas(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut df = self.df.write();
        let dfr = &mut *df; // Lock guard isn't Send, but mut ref is.
        py.enter_polars_ok(|| dfr.as_single_chunk_par())?;
        let df = RwLockWriteGuard::downgrade(df);
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;
            let cat_columns = df
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
            let rbs = df
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
    #[pyo3(signature = (requested_schema))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let mut df = self.df.write();
        let dfr = &mut *df; // Lock guard isn't Send, but mut ref is.
        py.enter_polars_ok(|| dfr.as_single_chunk_par())?;
        let df = RwLockWriteGuard::downgrade(df);
        dataframe_to_stream(&df, py)
    }
}
