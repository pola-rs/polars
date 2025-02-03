use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use pyo3::IntoPyObjectExt;

use super::PySeries;
use crate::error::PyPolarsErr;
use crate::interop;
use crate::interop::arrow::to_py::series_to_stream;
use crate::prelude::*;

#[pymethods]
impl PySeries {
    /// Convert this Series to a Python list.
    /// This operation copies data.
    pub fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let series = &self.series;

        fn to_list_recursive<'py>(py: Python<'py>, series: &Series) -> PyResult<Bound<'py, PyAny>> {
            let pylist = match series.dtype() {
                DataType::Boolean => PyList::new(py, series.bool().map_err(PyPolarsErr::from)?)?,
                DataType::UInt8 => PyList::new(py, series.u8().map_err(PyPolarsErr::from)?)?,
                DataType::UInt16 => PyList::new(py, series.u16().map_err(PyPolarsErr::from)?)?,
                DataType::UInt32 => PyList::new(py, series.u32().map_err(PyPolarsErr::from)?)?,
                DataType::UInt64 => PyList::new(py, series.u64().map_err(PyPolarsErr::from)?)?,
                DataType::Int8 => PyList::new(py, series.i8().map_err(PyPolarsErr::from)?)?,
                DataType::Int16 => PyList::new(py, series.i16().map_err(PyPolarsErr::from)?)?,
                DataType::Int32 => PyList::new(py, series.i32().map_err(PyPolarsErr::from)?)?,
                DataType::Int64 => PyList::new(py, series.i64().map_err(PyPolarsErr::from)?)?,
                DataType::Int128 => PyList::new(py, series.i128().map_err(PyPolarsErr::from)?)?,
                DataType::Float32 => PyList::new(py, series.f32().map_err(PyPolarsErr::from)?)?,
                DataType::Float64 => PyList::new(py, series.f64().map_err(PyPolarsErr::from)?)?,
                DataType::Categorical(_, _) | DataType::Enum(_, _) => PyList::new(
                    py,
                    series.categorical().map_err(PyPolarsErr::from)?.iter_str(),
                )?,
                #[cfg(feature = "object")]
                DataType::Object(_, _) => {
                    let v = PyList::empty(py);
                    for i in 0..series.len() {
                        let obj: Option<&ObjectValue> = series.get_object(i).map(|any| any.into());
                        v.append(obj)?;
                    }
                    v
                },
                DataType::List(_) => {
                    let v = PyList::empty(py);
                    let ca = series.list().map_err(PyPolarsErr::from)?;
                    for opt_s in ca.amortized_iter() {
                        match opt_s {
                            None => {
                                v.append(py.None())?;
                            },
                            Some(s) => {
                                let pylst = to_list_recursive(py, s.as_ref())?;
                                v.append(pylst)?;
                            },
                        }
                    }
                    v
                },
                DataType::Array(_, _) => {
                    let v = PyList::empty(py);
                    let ca = series.array().map_err(PyPolarsErr::from)?;
                    for opt_s in ca.amortized_iter() {
                        match opt_s {
                            None => {
                                v.append(py.None())?;
                            },
                            Some(s) => {
                                let pylst = to_list_recursive(py, s.as_ref())?;
                                v.append(pylst)?;
                            },
                        }
                    }
                    v
                },
                DataType::Date => {
                    let ca = series.date().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Time => {
                    let ca = series.time().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Datetime(_, _) => {
                    let ca = series.datetime().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Decimal(_, _) => {
                    let ca = series.decimal().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::String => {
                    let ca = series.str().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Struct(_) => {
                    let ca = series.struct_().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Duration(_) => {
                    let ca = series.duration().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Binary => {
                    let ca = series.binary().map_err(PyPolarsErr::from)?;
                    return Wrap(ca).into_bound_py_any(py);
                },
                DataType::Null => {
                    let null: Option<u8> = None;
                    let n = series.len();
                    let iter = std::iter::repeat(null).take(n);
                    use std::iter::{Repeat, Take};
                    struct NullIter {
                        iter: Take<Repeat<Option<u8>>>,
                        n: usize,
                    }
                    impl Iterator for NullIter {
                        type Item = Option<u8>;

                        fn next(&mut self) -> Option<Self::Item> {
                            self.iter.next()
                        }
                        fn size_hint(&self) -> (usize, Option<usize>) {
                            (self.n, Some(self.n))
                        }
                    }
                    impl ExactSizeIterator for NullIter {}

                    PyList::new(py, NullIter { iter, n })?
                },
                DataType::Unknown(_) => {
                    panic!("to_list not implemented for unknown")
                },
                DataType::BinaryOffset => {
                    unreachable!()
                },
            };
            Ok(pylist.into_any())
        }

        to_list_recursive(py, series)
    }

    /// Return the underlying Arrow array.
    #[allow(clippy::wrong_self_convention)]
    fn to_arrow(&mut self, py: Python, compat_level: PyCompatLevel) -> PyResult<PyObject> {
        self.rechunk(py, true)?;
        let pyarrow = py.import("pyarrow")?;

        interop::arrow::to_py::to_py_array(
            self.series.to_arrow(0, compat_level.0),
            &self.series.field().to_arrow(compat_level.0),
            &pyarrow,
        )
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &'py self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        series_to_stream(&self.series, py)
    }
}
