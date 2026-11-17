use polars::prelude::*;
use polars_arrow::datatypes::IntegerType;
use polars_arrow::record_batch::RecordBatch;
use polars_compute::cast::CastOptionsImpl;
use polars_utils::itertools::Itertools;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};

use super::PyDataFrame;
use crate::conversion::{ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::interop::arrow::to_py::batches_to_py_table;
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
        let row = df
            .columns()
            .iter()
            .map(|s| match s.dtype() {
                DataType::Object(_) => {
                    let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                    obj.into_py_any(py)
                },
                _ => Wrap(s.get(idx).unwrap()).into_py_any(py),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, row)
    }

    #[cfg(feature = "object")]
    pub fn row_tuples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let df = self.df.read();
        let mut rechunked;
        // Rechunk if random access would become rather expensive.
        // TODO: iterate over the chunks directly instead of using random access.
        let df = if df.max_n_chunks() > 16 {
            rechunked = df.clone();
            py.enter_polars_ok(|| rechunked.rechunk_mut_par())?;
            &rechunked
        } else {
            &df
        };
        let mut row = Vec::with_capacity(df.width());
        let rows = (0..df.height())
            .map(|idx| {
                row.clear();
                for c in df.columns() {
                    row.push(match c.dtype() {
                        DataType::Null => py.None(),
                        DataType::Object(_) => {
                            let obj: Option<&ObjectValue> = c.get_object(idx).map(|any| any.into());
                            obj.into_py_any(py)?
                        },
                        _ => {
                            // SAFETY: we are in bounds.
                            let av = unsafe { c.get_unchecked(idx) };
                            Wrap(av).into_py_any(py)?
                        },
                    });
                }
                PyTuple::new(py, &row)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, rows)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_arrow(&self, py: Python<'_>, compat_level: PyCompatLevel) -> PyResult<Py<PyAny>> {
        let df = self.aligned_for_export(py)?;
        let schema = df.schema().to_arrow(compat_level.0);
        let batches = df.try_iter_chunks(compat_level.0, true);
        batches_to_py_table(py, batches, schema)
    }

    /// Export to a `pyarrow.Table` whose Categorical and Enum columns have the signed
    /// dictionary keys pandas requires.
    ///
    /// Note this will give bad results for columns with dtype `pl.Object`,
    /// since those can't be converted correctly via PyArrow. The calling Python
    /// code should make sure these are not included.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_pandas(&self, py: Python) -> PyResult<Py<PyAny>> {
        let df = self.aligned_for_export(py)?;

        let mut schema = df.schema().to_arrow(CompatLevel::oldest());
        let dict_columns = df
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let is_enum = match c.dtype() {
                    DataType::Enum(_, _) => true,
                    DataType::Categorical(_, _) => false,
                    _ => return None,
                };
                let dtype = ArrowDataType::Dictionary(
                    IntegerType::Int64,
                    Box::new(ArrowDataType::LargeUtf8),
                    is_enum,
                );
                Some((i, dtype))
            })
            .collect_vec();
        for (i, dtype) in &dict_columns {
            schema.get_at_index_mut(*i).unwrap().1.dtype = dtype.clone();
        }

        let batch_schema = Arc::new(schema.clone());
        let batches = df
            .try_iter_chunks(CompatLevel::oldest(), true)
            .map(move |batch| {
                let batch = batch?;
                if dict_columns.is_empty() {
                    return Ok(batch);
                }
                let length = batch.len();
                let mut arrays = batch.into_arrays();
                for (i, dtype) in &dict_columns {
                    arrays[*i] = polars_compute::cast::cast(
                        &*arrays[*i],
                        dtype,
                        CastOptionsImpl::default(),
                    )?;
                }
                Ok(RecordBatch::new(length, batch_schema.clone(), arrays))
            });
        batches_to_py_table(py, batches, schema)
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let df = self.aligned_for_export(py)?;
        let stream = df.into_arrow_stream(CompatLevel::newest());
        PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
    }
}

impl PyDataFrame {
    /// This frame with aligned chunks, stored back so that a later export doesn't rechunk again.
    ///
    /// The clone is aligned outside the lock, as rechunking an Object column needs the GIL.
    fn aligned_for_export(&self, py: Python<'_>) -> PyResult<DataFrame> {
        let mut df = self.df.read().clone();
        if df.should_rechunk() {
            py.enter_polars_ok(|| {
                df.align_chunks_par();
            })?;
            *self.df.write() = df.clone();
        }
        Ok(df)
    }
}
