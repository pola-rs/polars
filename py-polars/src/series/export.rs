use num_traits::{Float, NumCast};
use numpy::PyArray1;
use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::conversion::chunked_array::{decimal_to_pyobject_iter, time_to_pyobject_iter};
use crate::error::PyPolarsErr;
use crate::prelude::{ObjectValue, *};
use crate::{arrow_interop, raise_err, PySeries};

#[pymethods]
impl PySeries {
    /// Convert this Series to a Python list.
    /// This operation copies data.
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
                    DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                        PyList::new(py, series.categorical().unwrap().iter_str())
                    },
                    #[cfg(feature = "object")]
                    DataType::Object(_, _) => {
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
                    DataType::String => {
                        let ca = series.str().unwrap();
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
                    DataType::BinaryOffset => {
                        unreachable!()
                    },
                };
                pylist.to_object(py)
            }

            let pylist = to_list_recursive(py, series);
            pylist.to_object(py)
        })
    }

    /// Return the underlying Arrow array.
    #[allow(clippy::wrong_self_convention)]
    fn to_arrow(&mut self) -> PyResult<PyObject> {
        self.rechunk(true);
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;

            arrow_interop::to_py::to_py_array(self.series.to_arrow(0, false), py, pyarrow)
        })
    }

    /// Convert this Series to a NumPy ndarray.
    ///
    /// This method will copy data - numeric types without null values should
    /// be handled on the Python side in a zero-copy manner.
    ///
    /// This method will cast integers to floats so that `null = np.nan`.
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        use DataType::*;
        let s = &self.series;
        let out = match s.dtype() {
            Int8 => numeric_series_to_numpy::<Int8Type, f32>(py, s),
            Int16 => numeric_series_to_numpy::<Int16Type, f32>(py, s),
            Int32 => numeric_series_to_numpy::<Int32Type, f64>(py, s),
            Int64 => numeric_series_to_numpy::<Int64Type, f64>(py, s),
            UInt8 => numeric_series_to_numpy::<UInt8Type, f32>(py, s),
            UInt16 => numeric_series_to_numpy::<UInt16Type, f32>(py, s),
            UInt32 => numeric_series_to_numpy::<UInt32Type, f64>(py, s),
            UInt64 => numeric_series_to_numpy::<UInt64Type, f64>(py, s),
            Float32 => numeric_series_to_numpy::<Float32Type, f32>(py, s),
            Float64 => numeric_series_to_numpy::<Float64Type, f64>(py, s),
            Boolean => {
                let ca = s.bool().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                np_arr.into_py(py)
            },
            Date => date_series_to_numpy(py, s),
            Datetime(_, _) | Duration(_) => temporal_series_to_numpy(py, s),
            Time => {
                let ca = s.time().unwrap();
                let iter = time_to_pyobject_iter(py, ca);
                let np_arr = PyArray1::from_iter(py, iter.map(|v| v.into_py(py)));
                np_arr.into_py(py)
            },
            String => {
                let ca = s.str().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                np_arr.into_py(py)
            },
            Binary => {
                let ca = s.binary().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                np_arr.into_py(py)
            },
            Categorical(_, _) | Enum(_, _) => {
                let ca = s.categorical().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.iter_str().map(|s| s.into_py(py)));
                np_arr.into_py(py)
            },
            Decimal(_, _) => {
                let ca = s.decimal().unwrap();
                let iter = decimal_to_pyobject_iter(py, ca);
                let np_arr = PyArray1::from_iter(py, iter.map(|v| v.into_py(py)));
                np_arr.into_py(py)
            },
            #[cfg(feature = "object")]
            Object(_, _) => {
                let ca = s
                    .as_any()
                    .downcast_ref::<ObjectChunked<ObjectValue>>()
                    .unwrap();
                let np_arr =
                    PyArray1::from_iter(py, ca.into_iter().map(|opt_v| opt_v.to_object(py)));
                np_arr.into_py(py)
            },
            Null => {
                let n = s.len();
                let np_arr = PyArray1::from_iter(py, std::iter::repeat(f32::NAN).take(n));
                np_arr.into_py(py)
            },
            dt => {
                raise_err!(
                    format!("`to_numpy` not supported for dtype {dt:?}"),
                    ComputeError
                );
            },
        };
        Ok(out)
    }
}
/// Convert numeric types to f32 or f64 with NaN representing a null value
fn numeric_series_to_numpy<T, U>(py: Python, s: &Series) -> PyObject
where
    T: PolarsNumericType,
    U: Float + numpy::Element,
{
    let ca: &ChunkedArray<T> = s.as_ref().as_ref();
    let mapper = |opt_v: Option<T::Native>| match opt_v {
        Some(v) => NumCast::from(v).unwrap(),
        None => U::nan(),
    };
    let np_arr = PyArray1::from_iter(py, ca.iter().map(mapper));
    np_arr.into_py(py)
}
/// Convert dates directly to i64 with i64::MIN representing a null value
fn date_series_to_numpy(py: Python, s: &Series) -> PyObject {
    let s_phys = s.to_physical_repr();
    let ca = s_phys.i32().unwrap();
    let mapper = |opt_v: Option<i32>| match opt_v {
        Some(v) => v as i64,
        None => i64::MIN,
    };
    let np_arr = PyArray1::from_iter(py, ca.iter().map(mapper));
    np_arr.into_py(py)
}
/// Convert datetimes and durations with i64::MIN representing a null value
fn temporal_series_to_numpy(py: Python, s: &Series) -> PyObject {
    let s_phys = s.to_physical_repr();
    let ca = s_phys.i64().unwrap();
    let np_arr = PyArray1::from_iter(py, ca.iter().map(|v| v.unwrap_or(i64::MIN)));
    np_arr.into_py(py)
}
