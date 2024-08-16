use ndarray::IntoDimension;
use num_traits::{Float, NumCast};
use numpy::npyffi::flags;
use numpy::{Element, PyArray1};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PySlice;

use super::to_numpy_df::df_to_numpy;
use super::utils::{
    create_borrowed_np_array, dtype_supports_view, polars_dtype_to_np_temporal_dtype,
    reshape_numpy_array, series_contains_null,
};
use crate::conversion::chunked_array::{decimal_to_pyobject_iter, time_to_pyobject_iter};
use crate::conversion::ObjectValue;
use crate::series::PySeries;

#[pymethods]
impl PySeries {
    /// Convert this Series to a NumPy ndarray.
    ///
    /// This method copies data only when necessary. Set `allow_copy` to raise an error if copy
    /// is required. Set `writable` to make sure the resulting array is writable, possibly requiring
    /// copying the data.
    fn to_numpy(&self, py: Python, writable: bool, allow_copy: bool) -> PyResult<PyObject> {
        series_to_numpy(py, &self.series, writable, allow_copy)
    }

    /// Create a view of the data as a NumPy ndarray.
    ///
    /// WARNING: The resulting view will show the underlying value for nulls,
    /// which may be any value. The caller is responsible for handling nulls
    /// appropriately.
    fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        let (view, _) = try_series_to_numpy_view(py, &self.series, true, false)?;
        Some(view)
    }
}

/// Convert a Series to a NumPy ndarray.
pub(super) fn series_to_numpy(
    py: Python,
    s: &Series,
    writable: bool,
    allow_copy: bool,
) -> PyResult<PyObject> {
    if s.is_empty() {
        // Take this path to ensure a writable array.
        // This does not actually copy data for an empty Series.
        return Ok(series_to_numpy_with_copy(py, s, true));
    }
    if let Some((mut arr, writable_flag)) = try_series_to_numpy_view(py, s, false, allow_copy) {
        if writable && !writable_flag {
            if !allow_copy {
                return Err(PyRuntimeError::new_err(
                    "copy not allowed: cannot create a writable array without copying data",
                ));
            }
            arr = arr.call_method0(py, intern!(py, "copy"))?;
        }
        return Ok(arr);
    }

    if !allow_copy {
        return Err(PyRuntimeError::new_err(
            "copy not allowed: cannot convert to a NumPy array without copying data",
        ));
    }

    Ok(series_to_numpy_with_copy(py, s, writable))
}

/// Create a NumPy view of the given Series.
fn try_series_to_numpy_view(
    py: Python,
    s: &Series,
    allow_nulls: bool,
    allow_rechunk: bool,
) -> Option<(PyObject, bool)> {
    if !dtype_supports_view(s.dtype()) {
        return None;
    }
    if !allow_nulls && series_contains_null(s) {
        return None;
    }
    let (s_owned, writable_flag) = handle_chunks(s, allow_rechunk)?;

    let array = series_to_numpy_view_recursive(py, s_owned, writable_flag);
    Some((array, writable_flag))
}
/// Rechunk the Series if required.
///
/// NumPy arrays are always contiguous, so we may have to rechunk before creating a view.
/// If we do so, we can flag the resulting array as writable.
fn handle_chunks(s: &Series, allow_rechunk: bool) -> Option<(Series, bool)> {
    let is_chunked = s.n_chunks() > 1;
    match (is_chunked, allow_rechunk) {
        (true, false) => None,
        (true, true) => Some((s.rechunk(), true)),
        (false, _) => Some((s.clone(), false)),
    }
}

/// Create a NumPy view of the given Series without checking for data types, chunks, or nulls.
fn series_to_numpy_view_recursive(py: Python, s: Series, writable: bool) -> PyObject {
    debug_assert!(s.n_chunks() == 1);
    match s.dtype() {
        dt if dt.is_numeric() => numeric_series_to_numpy_view(py, s, writable),
        DataType::Datetime(_, _) | DataType::Duration(_) => {
            temporal_series_to_numpy_view(py, s, writable)
        },
        DataType::Array(_, _) => array_series_to_numpy_view(py, &s, writable),
        _ => panic!("invalid data type"),
    }
}
/// Create a NumPy view of a numeric Series.
fn numeric_series_to_numpy_view(py: Python, s: Series, writable: bool) -> PyObject {
    let dims = [s.len()].into_dimension();
    with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
        let np_dtype = <$T as PolarsNumericType>::Native::get_dtype_bound(py);
        let ca: &ChunkedArray<$T> = s.unpack::<$T>().unwrap();
        let flags = if writable {
            flags::NPY_ARRAY_FARRAY
        } else {
            flags::NPY_ARRAY_FARRAY_RO
        };

        let slice = ca.data_views().next().unwrap();

        unsafe {
            create_borrowed_np_array::<_>(
                py,
                np_dtype,
                dims,
                flags,
                slice.as_ptr() as _,
                PySeries::from(s).into_py(py), // Keep the Series memory alive.,
            )
        }
    })
}
/// Create a NumPy view of a Datetime or Duration Series.
fn temporal_series_to_numpy_view(py: Python, s: Series, writable: bool) -> PyObject {
    let np_dtype = polars_dtype_to_np_temporal_dtype(py, s.dtype());

    let phys = s.to_physical_repr();
    let ca = phys.i64().unwrap();
    let slice = ca.data_views().next().unwrap();
    let dims = [s.len()].into_dimension();
    let flags = if writable {
        flags::NPY_ARRAY_FARRAY
    } else {
        flags::NPY_ARRAY_FARRAY_RO
    };

    unsafe {
        create_borrowed_np_array::<_>(
            py,
            np_dtype,
            dims,
            flags,
            slice.as_ptr() as _,
            PySeries::from(s).into_py(py), // Keep the Series memory alive.,
        )
    }
}
/// Create a NumPy view of an Array Series.
fn array_series_to_numpy_view(py: Python, s: &Series, writable: bool) -> PyObject {
    let ca = s.array().unwrap();
    let s_inner = ca.get_inner();
    let np_array_flat = series_to_numpy_view_recursive(py, s_inner, writable);

    // Reshape to the original shape.
    let DataType::Array(_, width) = s.dtype() else {
        unreachable!()
    };
    reshape_numpy_array(py, np_array_flat, ca.len(), *width)
}

/// Convert a Series to a NumPy ndarray, copying data in the process.
///
/// This method will cast integers to floats so that `null = np.nan`.
fn series_to_numpy_with_copy(py: Python, s: &Series, writable: bool) -> PyObject {
    use DataType::*;
    match s.dtype() {
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
        Struct(_) => {
            let ca = s.struct_().unwrap();
            let df = ca.clone().unnest();
            df_to_numpy(py, &df, IndexOrder::Fortran, writable, true).unwrap()
        },
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
        Unknown(_) | BinaryOffset => unreachable!(),
    }
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
            .call_method1(py, intern!(py, "__getitem__"), (slice,))
            .unwrap()
    });

    PyArray1::from_iter_bound(py, values).into_py(py)
}
/// Convert arrays by flattening first, converting the flat Series, and then reshaping.
fn array_series_to_numpy(py: Python, s: &Series, writable: bool) -> PyObject {
    let ca = s.array().unwrap();
    let s_inner = ca.get_inner();
    let np_array_flat = series_to_numpy_with_copy(py, &s_inner, writable);

    // Reshape to the original shape.
    let DataType::Array(_, width) = s.dtype() else {
        unreachable!()
    };
    reshape_numpy_array(py, np_array_flat, ca.len(), *width)
}
