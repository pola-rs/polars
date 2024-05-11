use num_traits::{Float, NumCast};
use numpy::PyArray1;
use polars_core::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::conversion::chunked_array::{decimal_to_pyobject_iter, time_to_pyobject_iter};
use crate::error::PyPolarsErr;
use crate::prelude::*;
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
                    DataType::Boolean => PyList::new_bound(py, series.bool().unwrap()),
                    DataType::UInt8 => PyList::new_bound(py, series.u8().unwrap()),
                    DataType::UInt16 => PyList::new_bound(py, series.u16().unwrap()),
                    DataType::UInt32 => PyList::new_bound(py, series.u32().unwrap()),
                    DataType::UInt64 => PyList::new_bound(py, series.u64().unwrap()),
                    DataType::Int8 => PyList::new_bound(py, series.i8().unwrap()),
                    DataType::Int16 => PyList::new_bound(py, series.i16().unwrap()),
                    DataType::Int32 => PyList::new_bound(py, series.i32().unwrap()),
                    DataType::Int64 => PyList::new_bound(py, series.i64().unwrap()),
                    DataType::Float32 => PyList::new_bound(py, series.f32().unwrap()),
                    DataType::Float64 => PyList::new_bound(py, series.f64().unwrap()),
                    DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                        PyList::new_bound(py, series.categorical().unwrap().iter_str())
                    },
                    #[cfg(feature = "object")]
                    DataType::Object(_, _) => {
                        let v = PyList::empty_bound(py);
                        for i in 0..series.len() {
                            let obj: Option<&ObjectValue> =
                                series.get_object(i).map(|any| any.into());
                            let val = obj.to_object(py);

                            v.append(val).unwrap();
                        }
                        v
                    },
                    DataType::List(_) => {
                        let v = PyList::empty_bound(py);
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
                        let v = PyList::empty_bound(py);
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

                        PyList::new_bound(py, NullIter { iter, n })
                    },
                    DataType::Unknown(_) => {
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
            let pyarrow = py.import_bound("pyarrow")?;

            arrow_interop::to_py::to_py_array(self.series.to_arrow(0, false), py, &pyarrow)
        })
    }

    /// Convert this Series to a NumPy ndarray.
    ///
    /// This method copies data only when necessary. Set `allow_copy` to raise an error if copy
    /// is required. Set `writable` to make sure the resulting array is writable, possibly requiring
    /// copying the data.
    fn to_numpy(&self, py: Python, allow_copy: bool, writable: bool) -> PyResult<PyObject> {
        let is_empty = self.series.is_empty();

        if self.series.null_count() == 0 {
            if let Some(mut arr) = self.to_numpy_view(py) {
                if writable || is_empty {
                    if !allow_copy && !is_empty {
                        return Err(PyValueError::new_err(
                            "cannot return a zero-copy writable array",
                        ));
                    }
                    arr = arr.call_method0(py, intern!(py, "copy"))?;
                }
                return Ok(arr);
            }
        }

        if !allow_copy & !is_empty {
            return Err(PyValueError::new_err("cannot return a zero-copy array"));
        }

        series_to_numpy_with_copy(py, &self.series)
    }
}

/// Convert a Series to a NumPy ndarray, copying data in the process.
///
/// This method will cast integers to floats so that `null = np.nan`.
fn series_to_numpy_with_copy(py: Python, s: &Series) -> PyResult<PyObject> {
    use DataType::*;
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
        Boolean => boolean_series_to_numpy(py, s),
        Date => date_series_to_numpy(py, s),
        Datetime(tu, _) => {
            use numpy::datetime::{units, Datetime};
            match tu {
                TimeUnit::Milliseconds => {
                    temporal_series_to_numpy::<Datetime<units::Milliseconds>>(py, s)
                },
                TimeUnit::Microseconds => {
                    temporal_series_to_numpy::<Datetime<units::Microseconds>>(py, s)
                },
                TimeUnit::Nanoseconds => {
                    temporal_series_to_numpy::<Datetime<units::Nanoseconds>>(py, s)
                },
            }
        },
        Duration(tu) => {
            use numpy::datetime::{units, Timedelta};
            match tu {
                TimeUnit::Milliseconds => {
                    temporal_series_to_numpy::<Timedelta<units::Milliseconds>>(py, s)
                },
                TimeUnit::Microseconds => {
                    temporal_series_to_numpy::<Timedelta<units::Microseconds>>(py, s)
                },
                TimeUnit::Nanoseconds => {
                    temporal_series_to_numpy::<Timedelta<units::Nanoseconds>>(py, s)
                },
            }
        },
        Time => {
            let ca = s.time().unwrap();
            let iter = time_to_pyobject_iter(py, ca);
            let np_arr = PyArray1::from_iter_bound(py, iter.map(|v| v.into_py(py)));
            np_arr.into_py(py)
        },
        String => {
            let ca = s.str().unwrap();
            let np_arr = PyArray1::from_iter_bound(py, ca.iter().map(|s| s.into_py(py)));
            np_arr.into_py(py)
        },
        Binary => {
            let ca = s.binary().unwrap();
            let np_arr = PyArray1::from_iter_bound(py, ca.iter().map(|s| s.into_py(py)));
            np_arr.into_py(py)
        },
        Categorical(_, _) | Enum(_, _) => {
            let ca = s.categorical().unwrap();
            let np_arr = PyArray1::from_iter_bound(py, ca.iter_str().map(|s| s.into_py(py)));
            np_arr.into_py(py)
        },
        Decimal(_, _) => {
            let ca = s.decimal().unwrap();
            let iter = decimal_to_pyobject_iter(py, ca);
            let np_arr = PyArray1::from_iter_bound(py, iter.map(|v| v.into_py(py)));
            np_arr.into_py(py)
        },
        #[cfg(feature = "object")]
        Object(_, _) => {
            let ca = s
                .as_any()
                .downcast_ref::<ObjectChunked<ObjectValue>>()
                .unwrap();
            let np_arr =
                PyArray1::from_iter_bound(py, ca.into_iter().map(|opt_v| opt_v.to_object(py)));
            np_arr.into_py(py)
        },
        Null => {
            let n = s.len();
            let np_arr = PyArray1::from_iter_bound(py, std::iter::repeat(f32::NAN).take(n));
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

/// Convert numeric types to f32 or f64 with NaN representing a null value.
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
    let np_arr = PyArray1::from_iter_bound(py, ca.iter().map(mapper));
    np_arr.into_py(py)
}
/// Convert booleans to u8 if no nulls are present, otherwise convert to objects.
fn boolean_series_to_numpy(py: Python, s: &Series) -> PyObject {
    let ca = s.bool().unwrap();
    if s.null_count() == 0 {
        let values = ca.into_no_null_iter();
        PyArray1::<bool>::from_iter_bound(py, values).into_py(py)
    } else {
        let values = ca.iter().map(|opt_v| opt_v.into_py(py));
        PyArray1::from_iter_bound(py, values).into_py(py)
    }
}
/// Convert dates directly to i64 with i64::MIN representing a null value.
fn date_series_to_numpy(py: Python, s: &Series) -> PyObject {
    use numpy::datetime::{units, Datetime};

    let s_phys = s.to_physical_repr();
    let ca = s_phys.i32().unwrap();

    if s.null_count() == 0 {
        let mapper = |v: i32| (v as i64).into();
        let values = ca.into_no_null_iter().map(mapper);
        PyArray1::<Datetime<units::Days>>::from_iter_bound(py, values).into_py(py)
    } else {
        let mapper = |opt_v: Option<i32>| {
            match opt_v {
                Some(v) => v as i64,
                None => i64::MIN,
            }
            .into()
        };
        let values = ca.iter().map(mapper);
        PyArray1::<Datetime<units::Days>>::from_iter_bound(py, values).into_py(py)
    }
}
/// Convert datetimes and durations with i64::MIN representing a null value.
fn temporal_series_to_numpy<T>(py: Python, s: &Series) -> PyObject
where
    T: From<i64> + numpy::Element,
{
    let s_phys = s.to_physical_repr();
    let ca = s_phys.i64().unwrap();
    let iter = ca.iter().map(|v| v.unwrap_or(i64::MIN).into());
    PyArray1::<T>::from_iter_bound(py, iter).into_py(py)
}
