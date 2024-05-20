use num_traits::{Float, NumCast};
use numpy::PyArray1;
use polars_core::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};

use crate::conversion::chunked_array::{decimal_to_pyobject_iter, time_to_pyobject_iter};
use crate::error::PyPolarsErr;
use crate::interop::numpy::to_py::{reshape_numpy_array, try_series_to_numpy_view};
use crate::prelude::*;
use crate::{interop, raise_err, PySeries};

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

            interop::arrow::to_py::to_py_array(self.series.to_arrow(0, false), py, &pyarrow)
        })
    }

    /// Convert this Series to a NumPy ndarray.
    ///
    /// This method copies data only when necessary. Set `allow_copy` to raise an error if copy
    /// is required. Set `writable` to make sure the resulting array is writable, possibly requiring
    /// copying the data.
    fn to_numpy(&self, py: Python, writable: bool, allow_copy: bool) -> PyResult<PyObject> {
        series_to_numpy(py, &self.series, writable, allow_copy)
    }
}

/// Convert a Series to a NumPy ndarray.
fn series_to_numpy(py: Python, s: &Series, writable: bool, allow_copy: bool) -> PyResult<PyObject> {
    if s.is_empty() {
        // Take this path to ensure a writable array.
        // This does not actually copy data for empty Series.
        return series_to_numpy_with_copy(py, s, true);
    }
    if let Some((mut arr, writable_flag)) = try_series_to_numpy_view(py, s, false, allow_copy) {
        if writable && !writable_flag {
            if !allow_copy {
                return Err(PyValueError::new_err(
                    "cannot return a zero-copy writable array",
                ));
            }
            arr = arr.call_method0(py, intern!(py, "copy"))?;
        }
        return Ok(arr);
    }

    if !allow_copy {
        return Err(PyValueError::new_err("cannot return a zero-copy array"));
    }

    series_to_numpy_with_copy(py, s, writable)
}

/// Convert a Series to a NumPy ndarray, copying data in the process.
///
/// This method will cast integers to floats so that `null = np.nan`.
fn series_to_numpy_with_copy(py: Python, s: &Series, writable: bool) -> PyResult<PyObject> {
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
            let values = time_to_pyobject_iter(ca).map(|v| v.into_py(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        String => {
            let ca = s.str().unwrap();
            let values = ca.iter().map(|s| s.into_py(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        Binary => {
            let ca = s.binary().unwrap();
            let values = ca.iter().map(|s| s.into_py(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        Categorical(_, _) | Enum(_, _) => {
            let ca = s.categorical().unwrap();
            let values = ca.iter_str().map(|s| s.into_py(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        Decimal(_, _) => {
            let ca = s.decimal().unwrap();
            let values = decimal_to_pyobject_iter(py, ca).map(|v| v.into_py(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        List(_) => list_series_to_numpy(py, s, writable),
        Array(_, _) => array_series_to_numpy(py, s, writable),
        #[cfg(feature = "object")]
        Object(_, _) => {
            let ca = s
                .as_any()
                .downcast_ref::<ObjectChunked<ObjectValue>>()
                .unwrap();
            let values = ca.iter().map(|v| v.to_object(py));
            PyArray1::from_iter_bound(py, values).into_py(py)
        },
        Null => {
            let n = s.len();
            let values = std::iter::repeat(f32::NAN).take(n);
            PyArray1::from_iter_bound(py, values).into_py(py)
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
    T::Native: numpy::Element,
    U: Float + numpy::Element,
{
    let ca: &ChunkedArray<T> = s.as_ref().as_ref();
    if s.null_count() == 0 {
        let values = ca.into_no_null_iter();
        PyArray1::<T::Native>::from_iter_bound(py, values).into_py(py)
    } else {
        let mapper = |opt_v: Option<T::Native>| match opt_v {
            Some(v) => NumCast::from(v).unwrap(),
            None => U::nan(),
        };
        let values = ca.iter().map(mapper);
        PyArray1::from_iter_bound(py, values).into_py(py)
    }
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
    let values = ca.iter().map(|v| v.unwrap_or(i64::MIN).into());
    PyArray1::<T>::from_iter_bound(py, values).into_py(py)
}
/// Convert arrays by flattening first, converting the flat Series, and then reshaping.
fn array_series_to_numpy(py: Python, s: &Series, writable: bool) -> PyObject {
    let ca = s.array().unwrap();
    let s_inner = ca.get_inner();
    let np_array_flat = series_to_numpy_with_copy(py, &s_inner, writable).unwrap();

    // Reshape to the original shape.
    let DataType::Array(_, width) = s.dtype() else {
        unreachable!()
    };
    reshape_numpy_array(py, np_array_flat, ca.len(), *width)
}
/// Convert lists by flattening first, converting the flat Series, and then splitting by offsets.
fn list_series_to_numpy(py: Python, s: &Series, writable: bool) -> PyObject {
    let ca = s.list().unwrap();
    let s_inner = ca.get_inner();

    let np_array_flat = series_to_numpy(py, &s_inner, writable, true).unwrap();

    // Split the NumPy array into subarrays by offset.
    // TODO: Downcast the NumPy array to Rust and split without calling into Python.
    let mut offsets = ca.iter_offsets().map(|o| isize::try_from(o).unwrap());
    let mut prev_offset = offsets.next().unwrap();
    let values = offsets.map(|current_offset| {
        let slice = PySlice::new_bound(py, prev_offset, current_offset, 1);
        prev_offset = current_offset;
        np_array_flat
            .call_method1(py, "__getitem__", (slice,))
            .unwrap()
    });

    PyArray1::from_iter_bound(py, values).into_py(py)
}
