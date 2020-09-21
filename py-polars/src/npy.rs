use ndarray::IntoDimension;
use numpy::{
    npyffi::{self, flags, types::npy_intp},
    ToNpyDims, PY_ARRAY_API,
};
use numpy::{Element, PyArray1};
use polars::prelude::*;
use pyo3::prelude::*;
use std::{mem, ptr};

/// Create an empty numpy array arrows 64 byte alignment
pub fn aligned_array<T: Element>(py: Python<'_>, size: usize) -> (&PyArray1<T>, *mut T) {
    let mut buf: Vec<T> = Vec::with_capacity_aligned(size);
    unsafe { buf.set_len(size) }
    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();

    let dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];
    unsafe {
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            T::ffi_dtype() as i32,
            strides.as_ptr() as *mut _, // strides
            buffer_ptr as _,            // data
            mem::size_of::<T>() as i32, // itemsize
            flags::NPY_ARRAY_OUT_ARRAY, // flag
            ptr::null_mut(),            //obj
        );
        mem::forget(buf);
        (PyArray1::from_owned_ptr(py, ptr), buffer_ptr)
    }
}

pub unsafe fn vec_from_ptr<T>(ptr: usize, len: usize) -> Vec<T> {
    let ptr = ptr as *mut T;
    Vec::from_raw_parts(ptr, len, len)
}
