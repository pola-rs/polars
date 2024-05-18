use std::ffi::{c_int, c_void};

use ndarray::{Dim, Dimension, IntoDimension};
use numpy::npyffi::{flags, PyArrayObject};
use numpy::{
    npyffi, Element, IntoPyArray, PyArrayDescr, PyArrayDescrMethods, ToNpyDims, PY_ARRAY_API,
};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::series::PySeries;

pub(crate) unsafe fn create_borrowed_np_array<I>(
    py: Python,
    dtype: Bound<PyArrayDescr>,
    mut shape: Dim<I>,
    flags: c_int,
    data: *mut c_void,
    owner: PyObject,
) -> PyObject
where
    Dim<I>: Dimension + ToNpyDims,
{
    // See: https://numpy.org/doc/stable/reference/c-api/array.html
    let array = PY_ARRAY_API.PyArray_NewFromDescr(
        py,
        PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
        dtype.into_dtype_ptr(),
        shape.ndim_cint(),
        shape.as_dims_ptr(),
        // We don't provide strides, but provide flags that tell c/f-order
        std::ptr::null_mut(),
        data,
        flags,
        std::ptr::null_mut(),
    );

    // This keeps the memory alive
    let owner_ptr = owner.as_ptr();
    // SetBaseObject steals a reference
    // so we can forget.
    std::mem::forget(owner);
    PY_ARRAY_API.PyArray_SetBaseObject(py, array as *mut PyArrayObject, owner_ptr);

    Py::from_owned_ptr(py, array)
}

#[pymethods]
impl PySeries {
    /// Create a view of the data as a NumPy ndarray.
    ///
    /// WARNING: The resulting view will show the underlying value for nulls,
    /// which may be any value. The caller is responsible for handling nulls
    /// appropriately.
    pub fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        let (view, _) = try_series_to_numpy_view(py, &self.series, true, false)?;
        Some(view)
    }
}

/// Create a NumPy view of the given Series.
pub(crate) fn try_series_to_numpy_view(
    py: Python,
    s: &Series,
    allow_nulls: bool,
    allow_rechunk: bool,
) -> Option<(PyObject, bool)> {
    if !supports_view(s.dtype()) {
        return None;
    }
    if !allow_nulls && has_nulls(s) {
        return None;
    }
    let (s_owned, writable_flag) = handle_chunks(s, allow_rechunk)?;

    let array = series_to_numpy_view_recursive(py, s_owned, writable_flag);
    Some((array, writable_flag))
}
/// Returns whether the data type supports creating a NumPy view.
fn supports_view(dtype: &DataType) -> bool {
    match dtype {
        dt if dt.is_numeric() => true,
        DataType::Datetime(_, _) | DataType::Duration(_) => true,
        DataType::Array(inner, _) => supports_view(inner.as_ref()),
        _ => false,
    }
}
/// Returns whether the Series contains nulls at any level of nesting.
///
/// Of the nested types, only Array types are handled since only those are relevant for NumPy views.
fn has_nulls(s: &Series) -> bool {
    if s.null_count() > 0 {
        true
    } else if s.dtype().is_array() {
        let ca = s.array().unwrap();
        let s_inner = ca.get_inner();
        has_nulls(&s_inner)
    } else {
        false
    }
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
/// Get the NumPy temporal data type associated with the given Polars [`DataType`].
fn polars_dtype_to_np_temporal_dtype<'a>(
    py: Python<'a>,
    dtype: &DataType,
) -> Bound<'a, PyArrayDescr> {
    use numpy::datetime::{units, Datetime, Timedelta};
    match dtype {
        DataType::Datetime(TimeUnit::Milliseconds, _) => {
            Datetime::<units::Milliseconds>::get_dtype_bound(py)
        },
        DataType::Datetime(TimeUnit::Microseconds, _) => {
            Datetime::<units::Microseconds>::get_dtype_bound(py)
        },
        DataType::Datetime(TimeUnit::Nanoseconds, _) => {
            Datetime::<units::Nanoseconds>::get_dtype_bound(py)
        },
        DataType::Duration(TimeUnit::Milliseconds) => {
            Timedelta::<units::Milliseconds>::get_dtype_bound(py)
        },
        DataType::Duration(TimeUnit::Microseconds) => {
            Timedelta::<units::Microseconds>::get_dtype_bound(py)
        },
        DataType::Duration(TimeUnit::Nanoseconds) => {
            Timedelta::<units::Nanoseconds>::get_dtype_bound(py)
        },
        _ => panic!("only Datetime/Duration inputs supported, got {}", dtype),
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
/// Reshape the first dimension of a NumPy array to the given height and width.
pub(crate) fn reshape_numpy_array(
    py: Python,
    arr: PyObject,
    height: usize,
    width: usize,
) -> PyObject {
    let shape = arr
        .getattr(py, intern!(py, "shape"))
        .unwrap()
        .extract::<Vec<usize>>(py)
        .unwrap();

    if shape.len() == 1 {
        // In this case, we can avoid allocating a Vec.
        let new_shape = (height, width);
        arr.call_method1(py, intern!(py, "reshape"), new_shape)
            .unwrap()
    } else {
        let mut new_shape_vec = vec![height, width];
        for v in &shape[1..] {
            new_shape_vec.push(*v)
        }
        let new_shape = PyTuple::new_bound(py, new_shape_vec);
        arr.call_method1(py, intern!(py, "reshape"), new_shape)
            .unwrap()
    }
}

#[pymethods]
#[allow(clippy::wrong_self_convention)]
impl PyDataFrame {
    pub fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        if self.df.is_empty() {
            return None;
        }
        let first = self.df.get_columns().first().unwrap().dtype();
        // TODO: Support Datetime/Duration types
        if !first.is_numeric() {
            return None;
        }
        if !self
            .df
            .get_columns()
            .iter()
            .all(|s| s.null_count() == 0 && s.dtype() == first && s.chunks().len() == 1)
        {
            return None;
        }

        let owner = self.clone().into_py(py); // Keep the DataFrame memory alive.

        fn get_ptr<T>(py: Python, columns: &[Series], owner: PyObject) -> Option<PyObject>
        where
            T: PolarsNumericType,
            T::Native: Element,
        {
            let slices = columns
                .iter()
                .map(|s| {
                    let ca: &ChunkedArray<T> = s.unpack().unwrap();
                    ca.cont_slice().unwrap()
                })
                .collect::<Vec<_>>();

            let first = slices.first().unwrap();
            unsafe {
                let mut end_ptr = first.as_ptr().add(first.len());
                // Check if all arrays are from the same buffer
                let all_contiguous = slices[1..].iter().all(|slice| {
                    let valid = slice.as_ptr() == end_ptr;

                    end_ptr = slice.as_ptr().add(slice.len());

                    valid
                });

                if all_contiguous {
                    let start_ptr = first.as_ptr();
                    let dtype = T::Native::get_dtype_bound(py);
                    let dims = [first.len(), columns.len()].into_dimension();
                    Some(create_borrowed_np_array::<_>(
                        py,
                        dtype,
                        dims,
                        flags::NPY_ARRAY_FARRAY_RO,
                        start_ptr as _,
                        owner,
                    ))
                } else {
                    None
                }
            }
        }
        with_match_physical_numeric_polars_type!(first, |$T| {
            get_ptr::<$T>(py, self.df.get_columns(), owner)
        })
    }

    pub fn to_numpy(&self, py: Python, order: Wrap<IndexOrder>) -> Option<PyObject> {
        let mut st = None;
        for s in self.df.iter() {
            let dt_i = s.dtype();
            match st {
                None => st = Some(dt_i.clone()),
                Some(ref mut st) => {
                    *st = try_get_supertype(st, dt_i).ok()?;
                },
            }
        }
        let st = st?;

        #[rustfmt::skip]
        let pyarray = match st {
            DataType::UInt8 => self.df.to_ndarray::<UInt8Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Int8 => self.df.to_ndarray::<Int8Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::UInt16 => self.df.to_ndarray::<UInt16Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Int16 => self.df.to_ndarray::<Int16Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::UInt32 => self.df.to_ndarray::<UInt32Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::UInt64 => self.df.to_ndarray::<UInt64Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Int32 => self.df.to_ndarray::<Int32Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Int64 => self.df.to_ndarray::<Int64Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Float32 => self.df.to_ndarray::<Float32Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            DataType::Float64 => self.df.to_ndarray::<Float64Type>(order.0).ok()?.into_pyarray_bound(py).into_py(py),
            _ => return None,
        };
        Some(pyarray)
    }
}
