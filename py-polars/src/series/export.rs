use numpy::PyArray1;
use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::error::PyPolarsErr;
use crate::prelude::{ObjectValue, *};
use crate::{arrow_interop, raise_err, PySeries};

#[pymethods]
impl PySeries {
    #[allow(clippy::wrong_self_convention)]
    fn to_arrow(&mut self) -> PyResult<PyObject> {
        self.rechunk(true);
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;

            arrow_interop::to_py::to_py_array(self.series.to_arrow(0), py, pyarrow)
        })
    }

    /// For numeric types, this should only be called for Series with null types.
    /// Non-nullable types are handled with `view()`.
    /// This will cast to floats so that `None = np.nan`.
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        let s = &self.series;
        match s.dtype() {
            dt if dt.is_numeric() => {
                if s.bit_repr_is_large() {
                    let s = s.cast(&DataType::Float64).unwrap();
                    let ca = s.f64().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| opt_v.unwrap_or(f64::NAN)),
                    );
                    Ok(np_arr.into_py(py))
                } else {
                    let s = s.cast(&DataType::Float32).unwrap();
                    let ca = s.f32().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| opt_v.unwrap_or(f32::NAN)),
                    );
                    Ok(np_arr.into_py(py))
                }
            },
            DataType::Utf8 => {
                let ca = s.utf8().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            },
            DataType::Binary => {
                let ca = s.binary().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            },
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            },
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                let ca = s
                    .as_any()
                    .downcast_ref::<ObjectChunked<ObjectValue>>()
                    .unwrap();
                let np_arr =
                    PyArray1::from_iter(py, ca.into_iter().map(|opt_v| opt_v.to_object(py)));
                Ok(np_arr.into_py(py))
            },
            DataType::Null => {
                let n = s.len();
                let np_arr = PyArray1::from_iter(py, std::iter::repeat(f32::NAN).take(n));
                Ok(np_arr.into_py(py))
            },
            dt => {
                raise_err!(
                    format!("'to_numpy' not supported for dtype: {dt:?}"),
                    ComputeError
                );
            },
        }
    }

    pub fn to_list(&self) -> PyObject {
        Python::with_gil(|py| {
            let series = &self.series;

            fn to_list_recursive(py: Python, series: &Series) -> PyObject {
                let pylist = match series.dtype() {
                    DataType::Boolean => PyList::new(py, series.bool().unwrap()),
                    DataType::UInt8 => PyList::new(py, series.u8().unwrap()),
                    DataType::UInt16 => PyList::new(py, series.u16().unwrap()),
                    DataType::UInt32 => PyList::new(py, series.u32().unwrap()),
                    DataType::UInt64 => PyList::new(py, series.u64().unwrap()),
                    DataType::Int8 => PyList::new(py, series.i8().unwrap()),
                    DataType::Int16 => PyList::new(py, series.i16().unwrap()),
                    DataType::Int32 => PyList::new(py, series.i32().unwrap()),
                    DataType::Int64 => PyList::new(py, series.i64().unwrap()),
                    DataType::Float32 => PyList::new(py, series.f32().unwrap()),
                    DataType::Float64 => PyList::new(py, series.f64().unwrap()),
                    DataType::Categorical(_, _) => {
                        PyList::new(py, series.categorical().unwrap().iter_str())
                    },
                    #[cfg(feature = "object")]
                    DataType::Object(_) => {
                        let v = PyList::empty(py);
                        for i in 0..series.len() {
                            let obj: Option<&ObjectValue> =
                                series.get_object(i).map(|any| any.into());
                            let val = obj.to_object(py);

                            v.append(val).unwrap();
                        }
                        v
                    },
                    DataType::List(_) => {
                        let v = PyList::empty(py);
                        let ca = series.list().unwrap();
                        // SAFETY: unstable series never lives longer than the iterator.
                        for opt_s in unsafe { ca.amortized_iter() } {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                },
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                },
                            }
                        }
                        v
                    },
                    DataType::Array(_, _) => {
                        let v = PyList::empty(py);
                        let ca = series.array().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                },
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                },
                            }
                        }
                        v
                    },
                    DataType::Date => {
                        let ca = series.date().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Time => {
                        let ca = series.time().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Datetime(_, _) => {
                        let ca = series.datetime().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Decimal(_, _) => {
                        let ca = series.decimal().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Utf8 => {
                        let ca = series.utf8().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Struct(_) => {
                        let ca = series.struct_().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Duration(_) => {
                        let ca = series.duration().unwrap();
                        return Wrap(ca).to_object(py);
                    },
                    DataType::Binary => {
                        let ca = series.binary().unwrap();
                        return Wrap(ca).to_object(py);
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

                        PyList::new(py, NullIter { iter, n })
                    },
                    DataType::Unknown => {
                        panic!("to_list not implemented for unknown")
                    },
                };
                pylist.to_object(py)
            }

            let pylist = to_list_recursive(py, series);
            pylist.to_object(py)
        })
    }
}
