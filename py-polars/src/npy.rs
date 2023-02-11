use std::{mem, ptr};

use ndarray::IntoDimension;
use numpy::npyffi::types::npy_intp;
use numpy::npyffi::{self, flags};
use numpy::{Element, PyArray1, ToNpyDims, PY_ARRAY_API};
use polars_core::prelude::*;
use polars_core::utils::arrow::types::NativeType;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::PyPolarsErr;
use crate::raise_err;
use crate::series::PySeries;

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

    let mut dims = [len].into_dimension();
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

macro_rules! impl_ufuncs {
    ($name:ident, $type:ident, $unsafe_from_ptr_method:ident) => {
        #[pymethods]
        impl PySeries {
            // applies a ufunc by accepting a lambda out: ufunc(*args, out=out)
            // the out array is allocated in this method, send to Python and once the ufunc is applied
            // ownership is taken by Rust again to prevent memory leak.
            // if the ufunc fails, we first must take ownership back.
            pub fn $name(&self, lambda: &PyAny) -> PyResult<PySeries> {
                // numpy array object, and a *mut ptr
                Python::with_gil(|py| {
                    let size = self.len();
                    let (out_array, av) =
                        unsafe { aligned_array::<<$type as PolarsNumericType>::Native>(py, size) };

                    debug_assert_eq!(get_refcnt(out_array), 1);
                    // inserting it in a tuple increase the reference count by 1.
                    let args = PyTuple::new(py, &[out_array]);
                    debug_assert_eq!(get_refcnt(out_array), 2);

                    // whatever the result, we must take the leaked memory ownership back
                    let s = match lambda.call1(args) {
                        Ok(_) => {
                            // if this assert fails, the lambda has taken a reference to the object, so we must panic
                            // args and the lambda return have a reference, making a total of 3
                            assert_eq!(get_refcnt(out_array), 3);

                            let validity = self.series.chunks()[0].validity().cloned();
                            let ca = ChunkedArray::<$type>::new_from_owned_with_null_bitmap(
                                self.name(),
                                av,
                                validity,
                            );
                            PySeries::new(ca.into_series())
                        }
                        Err(e) => {
                            // return error information
                            return Err(e);
                        }
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

fn get_ptr<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
}

#[pymethods]
impl PySeries {
    pub fn get_ptr(&self) -> PyResult<usize> {
        let s = self.series.to_physical_repr();
        let arrays = s.chunks();
        if arrays.len() != 1 {
            let msg = "Only can take pointer, if the 'series' contains a single chunk";
            raise_err!(msg, ComputeError);
        }
        match s.dtype() {
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                // this one is quite useless as you need to know the offset
                // into the first byte.
                let (slice, start, _len) = arr.values().as_slice();
                if start == 0 {
                    Ok(slice.as_ptr() as usize)
                } else {
                    let msg = "Cannot take pointer boolean buffer as it is not perfectly aligned.";
                    raise_err!(msg, ComputeError);
                }
            }
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                get_ptr(ca)
            })),
            _ => {
                let msg = "Cannot take pointer of nested type";
                raise_err!(msg, ComputeError);
            }
        }
    }

    /// For numeric types, this should only be called for Series with null types.
    /// This will cast to floats so that `None = np.nan`
    pub fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        let s = &self.series;
        match s.dtype() {
            DataType::Utf8 => {
                let ca = s.utf8().unwrap();

                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            }
            dt if dt.is_numeric() => {
                if s.bit_repr_is_large() {
                    let s = s.cast(&DataType::Float64).unwrap();
                    let ca = s.f64().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| match opt_v {
                            Some(v) => v,
                            None => f64::NAN,
                        }),
                    );
                    Ok(np_arr.into_py(py))
                } else {
                    let s = s.cast(&DataType::Float32).unwrap();
                    let ca = s.f32().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| match opt_v {
                            Some(v) => v,
                            None => f32::NAN,
                        }),
                    );
                    Ok(np_arr.into_py(py))
                }
            }
            dt => {
                raise_err!(
                    format!("'to_numpy' not supported for dtype: {dt:?}"),
                    ComputeError
                );
            }
        }
    }
}
