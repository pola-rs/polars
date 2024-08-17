use std::{mem, ptr};

use ndarray::IntoDimension;
use numpy::npyffi::types::npy_intp;
use numpy::npyffi::{self, flags};
use numpy::{Element, PyArray1, PyArrayDescrMethods, ToNpyDims, PY_ARRAY_API};
use polars_core::prelude::*;
use polars_core::utils::arrow::types::NativeType;
use pyo3::prelude::*;
use pyo3::types::{PyNone, PyTuple};

use super::PySeries;

/// Create an empty numpy array arrows 64 byte alignment
///
/// # Safety
/// All elements in the array are non initialized
///
/// The array is also writable from Python.
unsafe fn aligned_array<T: Element + NativeType>(
    py: Python<'_>,
    size: usize,
) -> (Bound<'_, PyArray1<T>>, Vec<T>) {
    let mut buf = vec![T::default(); size];

    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();

    let mut dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];

    let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
        py,
        PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
        T::get_dtype_bound(py).into_dtype_ptr(),
        dims.ndim_cint(),
        dims.as_dims_ptr(),
        strides.as_ptr() as *mut _, // strides
        buffer_ptr as _,            // data
        flags::NPY_ARRAY_OUT_ARRAY, // flag
        ptr::null_mut(),            //obj
    );
    (
        Bound::from_owned_ptr(py, ptr)
            .downcast_into_exact::<PyArray1<T>>()
            .unwrap(),
        buf,
    )
}

/// Get reference counter for numpy arrays.
///   - For CPython: Get reference counter.
///   - For PyPy: Reference counters for a live PyPy object = refcnt + 2 << 60.
fn get_refcnt<T>(pyarray: &Bound<'_, PyArray1<T>>) -> isize {
    let refcnt = pyarray.get_refcnt();
    if refcnt >= (2 << 60) {
        refcnt - (2 << 60)
    } else {
        refcnt
    }
}

macro_rules! impl_ufuncs {
    ($name:ident, $type:ident, $unsafe_from_ptr_method:ident) => {
        #[pymethods]
        impl PySeries {
            // Applies a ufunc by accepting a lambda out: ufunc(*args, out=out).
            //
            // If allocate_out is true, the out array is allocated in this
            // method, send to Python and once the ufunc is applied ownership is
            // taken by Rust again to prevent memory leak. if the ufunc fails,
            // we first must take ownership back.
            //
            // If allocate_out is false, the out parameter to the lambda will be
            // None, meaning the ufunc will allocate memory itself. We will then
            // have to convert that NumPy array into a pl.Series.
            fn $name(&self, lambda: &Bound<PyAny>, allocate_out: bool) -> PyResult<PySeries> {
                // numpy array object, and a *mut ptr
                Python::with_gil(|py| {
                    if !allocate_out {
                        // We're not going to allocate the output array.
                        // Instead, we'll let the ufunc do it.
                        let result = lambda.call1((PyNone::get_bound(py),))?;
                        let series_factory =
                            PyModule::import_bound(py, "polars")?.getattr("Series")?;
                        return series_factory
                            .call((self.name(), result), None)?
                            .getattr("_s")?
                            .extract::<PySeries>();
                    }

                    let size = self.len();
                    let (out_array, av) =
                        unsafe { aligned_array::<<$type as PolarsNumericType>::Native>(py, size) };

                    debug_assert_eq!(get_refcnt(&out_array), 1);
                    // inserting it in a tuple increase the reference count by 1.
                    let args = PyTuple::new_bound(py, &[out_array.clone()]);
                    debug_assert_eq!(get_refcnt(&out_array), 2);

                    // whatever the result, we must take the leaked memory ownership back
                    let s = match lambda.call1(args) {
                        Ok(_) => {
                            // if this assert fails, the lambda has taken a reference to the object, so we must panic
                            // args and the lambda return have a reference, making a total of 3
                            assert!(get_refcnt(&out_array) <= 3);

                            let validity = self.series.chunks()[0].validity().cloned();
                            let ca =
                                ChunkedArray::<$type>::from_vec_validity(self.name(), av, validity);
                            PySeries::new(ca.into_series())
                        },
                        Err(e) => {
                            // return error information
                            return Err(e);
                        },
                    };

                    Ok(s)
                })
            }
        }
    };
}

impl_ufuncs!(apply_ufunc_f32, Float32Type, unsafe_from_ptr_f32);
impl_ufuncs!(apply_ufunc_f64, Float64Type, unsafe_from_ptr_f64);
impl_ufuncs!(apply_ufunc_u8, UInt8Type, unsafe_from_ptr_u8);
impl_ufuncs!(apply_ufunc_u16, UInt16Type, unsafe_from_ptr_u16);
impl_ufuncs!(apply_ufunc_u32, UInt32Type, unsafe_from_ptr_u32);
impl_ufuncs!(apply_ufunc_u64, UInt64Type, unsafe_from_ptr_u64);
impl_ufuncs!(apply_ufunc_i8, Int8Type, unsafe_from_ptr_i8);
impl_ufuncs!(apply_ufunc_i16, Int16Type, unsafe_from_ptr_i16);
impl_ufuncs!(apply_ufunc_i32, Int32Type, unsafe_from_ptr_i32);
impl_ufuncs!(apply_ufunc_i64, Int64Type, unsafe_from_ptr_i64);
