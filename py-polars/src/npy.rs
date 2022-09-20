use std::{mem, ptr};

use ndarray::IntoDimension;
use numpy::npyffi::types::npy_intp;
use numpy::npyffi::{self, flags};
use numpy::{Element, PyArray1, ToNpyDims, PY_ARRAY_API};
use polars_core::utils::arrow::types::NativeType;
use pyo3::prelude::*;

/// Create an empty numpy array arrows 64 byte alignment
///
/// # Safety
/// All elements in the array are non initialized
///
/// The array is also writable from Python.
pub unsafe fn aligned_array<T: Element + NativeType>(
    py: Python<'_>,
    size: usize,
) -> (&PyArray1<T>, Vec<T>) {
    let mut buf = vec![T::default(); size];

    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();

    let dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];

    let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
        py,
        PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
        T::get_dtype(py).into_dtype_ptr(),
        dims.ndim_cint(),
        dims.as_dims_ptr(),
        strides.as_ptr() as *mut _, // strides
        buffer_ptr as _,            // data
        flags::NPY_ARRAY_OUT_ARRAY, // flag
        ptr::null_mut(),            //obj
    );
    (PyArray1::from_owned_ptr(py, ptr), buf)
}

/// Get reference counter for numpy arrays.
///   - For CPython: Get reference counter.
///   - For PyPy: Reference counters for a live PyPy object = refcnt + 2 << 60.
pub fn get_refcnt<T>(pyarray: &PyArray1<T>) -> isize {
    let refcnt = pyarray.get_refcnt();
    if refcnt >= (2 << 60) {
        refcnt - (2 << 60)
    } else {
        refcnt
    }
}
