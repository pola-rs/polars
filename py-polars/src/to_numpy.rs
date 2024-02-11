use std::ffi::{c_int, c_void};

use ndarray::{Dim, Dimension, IntoDimension};
use numpy::npyffi::{flags, PyArrayObject};
use numpy::{npyffi, Element, IntoPyArray, ToNpyDims, PY_ARRAY_API};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::prelude::*;
use pyo3::{IntoPy, PyAny, PyObject, Python};

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::series::PySeries;

pub(crate) unsafe fn create_borrowed_np_array<T: NumericNative + Element, I>(
    py: Python,
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
        T::get_dtype(py).into_dtype_ptr(),
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

    let any: &PyAny = py.from_owned_ptr(array);
    any.into_py(py)
}

#[pymethods]
#[allow(clippy::wrong_self_convention)]
impl PySeries {
    pub fn to_numpy_view(&self, py: Python) -> Option<PyObject> {
        if self.series.null_count() != 0 || self.series.chunks().len() > 1 {
            return None;
        }

        match self.series.dtype() {
            dt if dt.is_numeric() => {
                let dims = [self.series.len()].into_dimension();
                // Object to the series keep the memory alive.
                let owner = self.clone().into_py(py);
                with_match_physical_numeric_polars_type!(self.series.dtype(), |$T| {
                            let ca: &ChunkedArray<$T> = self.series.unpack::<$T>().unwrap();
                            let slice = ca.cont_slice().unwrap();
                            unsafe { Some(create_borrowed_np_array::<<$T as PolarsNumericType>::Native, _>(
                                py,
                                dims,
                                flags::NPY_ARRAY_FARRAY_RO,
                                slice.as_ptr() as _,
                                owner,
                            )) }
                })
            },
            _ => None,
        }
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

        // Object to the dataframe keep the memory alive.
        let owner = self.clone().into_py(py);

        fn get_ptr<T: PolarsNumericType>(
            py: Python,
            columns: &[Series],
            owner: PyObject,
        ) -> Option<PyObject>
        where
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
                    let dims = [first.len(), columns.len()].into_dimension();
                    Some(create_borrowed_np_array::<T::Native, _>(
                        py,
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
            DataType::UInt8 => self.df.to_ndarray::<UInt8Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int8 => self.df.to_ndarray::<Int8Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::UInt16 => self.df.to_ndarray::<UInt16Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int16 => self.df.to_ndarray::<Int16Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::UInt32 => self.df.to_ndarray::<UInt32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::UInt64 => self.df.to_ndarray::<UInt64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int32 => self.df.to_ndarray::<Int32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int64 => self.df.to_ndarray::<Int64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Float32 => self.df.to_ndarray::<Float32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Float64 => self.df.to_ndarray::<Float64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            _ => return None,
        };
        Some(pyarray)
    }
}
