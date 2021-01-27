use ndarray::IntoDimension;
use numpy::{
    npyffi::{self, flags, types::npy_intp},
    ToNpyDims, PY_ARRAY_API,
};
use numpy::{Element, PyArray1};
use polars::chunked_array::builder::memory;
use polars::prelude::*;
use pyo3::prelude::*;
use std::any::Any;
use std::{mem, ptr};

/// Create an empty numpy array arrows 64 byte alignment
///
/// # Safety
/// All elements in the array are non initialized
///
/// The array is also writable from Python.
pub unsafe fn aligned_array<T: Element>(py: Python<'_>, size: usize) -> (&PyArray1<T>, *mut T) {
    let t_size = std::mem::size_of::<T>();
    let capacity = size * t_size;
    let ptr = memory::allocate_aligned(capacity).as_ptr() as *mut T;
    let mut buf = Vec::from_raw_parts(ptr, 0, capacity);
    buf.set_len(size);
    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();

    let dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];

    let ptr = PY_ARRAY_API.PyArray_New(
        PY_ARRAY_API.get_type_object(npyffi::NpyTypes::PyArray_Type),
        dims.ndim_cint(),
        dims.as_dims_ptr(),
        T::npy_type() as i32,
        strides.as_ptr() as *mut _, // strides
        buffer_ptr as _,            // data
        mem::size_of::<T>() as i32, // itemsize
        flags::NPY_ARRAY_OUT_ARRAY, // flag
        ptr::null_mut(),            //obj
    );
    mem::forget(buf);
    (PyArray1::from_owned_ptr(py, ptr), buffer_ptr)
}

pub unsafe fn vec_from_ptr<T>(ptr: usize, len: usize) -> Vec<T> {
    let ptr = ptr as *mut T;
    Vec::from_raw_parts(ptr, len, len)
}

pub fn series_to_numpy_compatible_vec(s: &Series) -> Box<dyn Any + Send> {
    // if has null values we use floats and np.nan to represent missing values
    macro_rules! impl_to_vec {
        ($ca:expr, $float_type:ty) => {{
            match $ca.cont_slice() {
                Ok(slice) => Box::new(Vec::from(slice)),
                Err(_) => {
                    if $ca.null_count() == 0 {
                        Box::new($ca.into_no_null_iter().collect::<Vec<_>>())
                    } else {
                        Box::new(
                            $ca.into_iter()
                                .map(|opt_v| match opt_v {
                                    Some(v) => v as $float_type,
                                    None => <$float_type>::NAN,
                                })
                                .collect::<Vec<_>>(),
                        )
                    }
                }
            }
        }};
    }

    // we use floating types only if null values
    match s.dtype() {
        DataType::Utf8 => Box::new(
            s.utf8()
                .unwrap()
                .into_iter()
                .map(|opt_s| opt_s.map(|s| s.to_string()))
                .collect::<Vec<_>>(),
        ),
        DataType::List(_) => Box::new(s.list().unwrap().into_iter().collect::<Vec<_>>()),
        DataType::UInt8 => impl_to_vec!(s.u8().unwrap(), f32),
        DataType::UInt16 => impl_to_vec!(s.u16().unwrap(), f32),
        DataType::UInt32 => impl_to_vec!(s.u32().unwrap(), f64),
        DataType::UInt64 => impl_to_vec!(s.u64().unwrap(), f64),
        DataType::Int8 => impl_to_vec!(s.i8().unwrap(), f32),
        DataType::Int16 => impl_to_vec!(s.i16().unwrap(), f32),
        DataType::Int32 => impl_to_vec!(s.i32().unwrap(), f64),
        DataType::Int64 => impl_to_vec!(s.i64().unwrap(), f64),
        DataType::Float32 => impl_to_vec!(s.f32().unwrap(), f32),
        DataType::Float64 => impl_to_vec!(s.f64().unwrap(), f64),
        DataType::Date32 => {
            impl_to_vec!(s.date32().unwrap(), f64)
        }
        DataType::Date64 => {
            impl_to_vec!(s.date64().unwrap(), f64)
        }
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            if ca.null_count() == 0 {
                Box::new(ca.into_no_null_iter().collect::<Vec<_>>())
            } else {
                Box::new(s.bool().unwrap().into_iter().collect::<Vec<_>>())
            }
        }
        dt => panic!(format!("{:?} not supported", dt)),
    }
}
