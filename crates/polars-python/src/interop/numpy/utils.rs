use std::ffi::{c_int, c_void};

use ndarray::{Dim, Dimension};
use numpy::npyffi::PyArrayObject;
use numpy::{npyffi, Element, PyArrayDescr, PyArrayDescrMethods, ToNpyDims, PY_ARRAY_API};
use polars_core::prelude::*;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Create a NumPy ndarray view of the data.
pub(super) unsafe fn create_borrowed_np_array<I>(
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

/// Returns whether the data type supports creating a NumPy view.
pub(super) fn dtype_supports_view(dtype: &DataType) -> bool {
    match dtype {
        dt if dt.is_numeric() => true,
        DataType::Datetime(_, _) | DataType::Duration(_) => true,
        DataType::Array(inner, _) => dtype_supports_view(inner.as_ref()),
        _ => false,
    }
}

/// Returns whether the Series contains nulls at any level of nesting.
///
/// Of the nested types, only Array types are handled since only those are relevant for NumPy views.
pub(super) fn series_contains_null(s: &Series) -> bool {
    if s.null_count() > 0 {
        true
    } else if let Ok(ca) = s.array() {
        let s_inner = ca.get_inner();
        series_contains_null(&s_inner)
    } else {
        false
    }
}

/// Reshape the first dimension of a NumPy array to the given height and width.
pub(super) fn reshape_numpy_array(
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

/// Get the NumPy temporal data type associated with the given Polars [`DataType`].
pub(super) fn polars_dtype_to_np_temporal_dtype<'a>(
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
