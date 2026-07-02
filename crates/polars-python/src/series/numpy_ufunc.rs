use arrow::array::{Array, PrimitiveArray};
use numpy::{PyArray1, PyArrayMethods};
use polars_buffer::{Buffer, SharedStorage};
use polars_core::prelude::*;
use polars_core::utils::arrow::types::NativeType;
use pyo3::prelude::*;
use pyo3::types::PyNone;

use super::PySeries;

macro_rules! impl_ufuncs {
    ($name:ident, $type:ident) => {
        #[pymethods]
        impl PySeries {
            // Applies a ufunc by accepting a lambda out: ufunc(*args, out=out).
            //
            // If allocate_out is true, the out array is allocated in this
            // method, sent to Python, and once the ufunc is applied a
            // ChunkedArray is created using the output array as backing buffer.
            //
            // If allocate_out is false, the out parameter to the lambda will be
            // None, meaning the ufunc will allocate memory itself. We will then
            // have to convert that NumPy array into a pl.Series.
            fn $name(&self, lambda: &Bound<PyAny>, allocate_out: bool) -> PyResult<PySeries> {
                Python::attach(|py| {
                    if !allocate_out {
                        let result = lambda.call1((PyNone::get(py),))?;
                        let series_factory = crate::py_modules::pl_series(py).bind(py);
                        return series_factory
                            .call((self.name(), result), None)?
                            .getattr("_s")?
                            .extract::<PySeries>()
                            .map_err(PyErr::from);
                    }

                    let size = self.len();
                    let out_array =
                        PyArray1::<<$type as PolarsNumericType>::Native>::zeros(py, [size], false);

                    match lambda.call1((&out_array,)) {
                        Ok(_) => {
                            let (name, validity) = {
                                let s = self.series.read();
                                (s.name().clone(), s.chunks()[0].validity().cloned())
                            };

                            // Create a Series backed by the numpy array's buffer.
                            // The Py<PyArray1<...>> owner keeps the array alive, preventing
                            // NumPy from freeing the buffer before the Series is dropped.
                            let owner = out_array.clone().unbind();
                            let ro = out_array.readonly();
                            let vals = ro.as_slice().unwrap();
                            let series = unsafe {
                                let storage = SharedStorage::from_slice_with_owner(vals, owner);
                                let buffer = Buffer::from_storage(storage);
                                let array = PrimitiveArray::new_unchecked(
                                    <<$type as PolarsNumericType>::Native as NativeType>::PRIMITIVE
                                        .into(),
                                    buffer,
                                    validity,
                                );
                                Series::from_arrow(name, array.to_boxed()).unwrap()
                            };
                            Ok(PySeries::new(series))
                        },
                        Err(e) => Err(e),
                    }
                })
            }
        }
    };
}

impl_ufuncs!(apply_ufunc_f32, Float32Type);
impl_ufuncs!(apply_ufunc_f64, Float64Type);
impl_ufuncs!(apply_ufunc_u8, UInt8Type);
impl_ufuncs!(apply_ufunc_u16, UInt16Type);
impl_ufuncs!(apply_ufunc_u32, UInt32Type);
impl_ufuncs!(apply_ufunc_u64, UInt64Type);
impl_ufuncs!(apply_ufunc_i8, Int8Type);
impl_ufuncs!(apply_ufunc_i16, Int16Type);
impl_ufuncs!(apply_ufunc_i32, Int32Type);
impl_ufuncs!(apply_ufunc_i64, Int64Type);
