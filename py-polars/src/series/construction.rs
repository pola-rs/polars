use std::borrow::Cow;

use numpy::{Element, PyArray1, PyArrayMethods};
use polars::export::arrow;
use polars::export::arrow::array::Array;
use polars::export::arrow::bitmap::MutableBitmap;
use polars::export::arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::utils::CustomIterTools;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::conversion::any_value::py_object_to_any_value;
use crate::conversion::{reinterpret_vec, Wrap};
use crate::error::PyPolarsErr;
use crate::interop::arrow::to_rust::array_to_rust;
use crate::prelude::ObjectValue;
use crate::PySeries;

// Init with numpy arrays.
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(
                py: Python,
                name: &str,
                array: &Bound<PyArray1<$type>>,
                _strict: bool,
            ) -> Self {
                mmap_numpy_array(py, name, array)
            }
        }
    };
}

init_method!(new_i8, i8);
init_method!(new_i16, i16);
init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);

fn mmap_numpy_array<T: Element + NativeType>(
    py: Python,
    name: &str,
    array: &Bound<PyArray1<T>>,
) -> PySeries {
    let vals = unsafe { array.as_slice().unwrap() };

    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(vals, array.to_object(py)) };
    Series::from_arrow(name, arr.to_boxed()).unwrap().into()
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_bool(py: Python, name: &str, array: &Bound<PyArray1<bool>>, _strict: bool) -> Self {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| Series::new(name, vals).into())
    }

    #[staticmethod]
    fn new_f32(py: Python, name: &str, array: &Bound<PyArray1<f32>>, nan_is_null: bool) -> Self {
        if nan_is_null {
            let array = array.readonly();
            let vals = array.as_slice().unwrap();
            let ca: Float32Chunked = vals
                .iter()
                .map(|&val| if f32::is_nan(val) { None } else { Some(val) })
                .collect_trusted();
            ca.with_name(name).into_series().into()
        } else {
            mmap_numpy_array(py, name, array)
        }
    }

    #[staticmethod]
    fn new_f64(py: Python, name: &str, array: &Bound<PyArray1<f64>>, nan_is_null: bool) -> Self {
        if nan_is_null {
            let array = array.readonly();
            let vals = array.as_slice().unwrap();
            let ca: Float64Chunked = vals
                .iter()
                .map(|&val| if f64::is_nan(val) { None } else { Some(val) })
                .collect_trusted();
            ca.with_name(name).into_series().into()
        } else {
            mmap_numpy_array(py, name, array)
        }
    }
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_opt_bool(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        let mut builder = BooleanChunkedBuilder::new(name, len);

        for res in values.iter()? {
            let value = res?;
            if value.is_none() {
                builder.append_null()
            } else {
                let v = value.extract::<bool>()?;
                builder.append_value(v)
            }
        }

        let ca = builder.finish();
        let s = ca.into_series();
        Ok(s.into())
    }
}

fn new_primitive<'a, T>(name: &str, values: &'a Bound<PyAny>, _strict: bool) -> PyResult<PySeries>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    T::Native: FromPyObject<'a>,
{
    let len = values.len()?;
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name, len);

    for res in values.iter()? {
        let value = res?;
        if value.is_none() {
            builder.append_null()
        } else {
            let v = value.extract::<T::Native>()?;
            builder.append_value(v)
        }
    }

    let ca = builder.finish();
    let s = ca.into_series();
    Ok(s.into())
}

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(name: &str, obj: &Bound<PyAny>, strict: bool) -> PyResult<Self> {
                new_primitive::<$type>(name, obj, strict)
            }
        }
    };
}

init_method_opt!(new_opt_u8, UInt8Type, u8);
init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_from_any_values(name: &str, values: &Bound<PyAny>, strict: bool) -> PyResult<Self> {
        let any_values_result = values
            .iter()?
            .map(|v| py_object_to_any_value(&(v?).as_borrowed(), strict))
            .collect::<PyResult<Vec<AnyValue>>>();
        let result = any_values_result.and_then(|avs| {
            let s = Series::from_any_values(name, avs.as_slice(), strict).map_err(|e| {
                PyTypeError::new_err(format!(
                    "{e}\n\nHint: Try setting `strict=False` to allow passing data with mixed types."
                ))
            })?;
            Ok(s.into())
        });

        // Fall back to Object type for non-strict construction.
        if !strict && result.is_err() {
            return Python::with_gil(|py| {
                let objects = values
                    .iter()?
                    .map(|v| {
                        let obj = ObjectValue {
                            inner: v?.to_object(py),
                        };
                        Ok(obj)
                    })
                    .collect::<PyResult<Vec<ObjectValue>>>()?;
                Ok(Self::new_object(py, name, objects, strict))
            });
        }

        result
    }

    #[staticmethod]
    fn new_from_any_values_and_dtype(
        name: &str,
        values: &Bound<PyAny>,
        dtype: Wrap<DataType>,
        strict: bool,
    ) -> PyResult<Self> {
        let any_values = values
            .iter()?
            .map(|v| py_object_to_any_value(&(v?).as_borrowed(), strict))
            .collect::<PyResult<Vec<AnyValue>>>()?;
        let s = Series::from_any_values_and_dtype(name, any_values.as_slice(), &dtype.0, strict)
            .map_err(|e| {
                PyTypeError::new_err(format!(
                    "{e}\n\nHint: Try setting `strict=False` to allow passing data with mixed types."
                ))
            })?;
        Ok(s.into())
    }

    #[staticmethod]
    fn new_str(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        let mut builder = StringChunkedBuilder::new(name, len);

        for res in values.iter()? {
            let value = res?;
            if value.is_none() {
                builder.append_null()
            } else {
                let v = value.extract::<Cow<str>>()?;
                builder.append_value(v)
            }
        }

        let ca = builder.finish();
        let s = ca.into_series();
        Ok(s.into())
    }

    #[staticmethod]
    fn new_binary(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        let mut builder = BinaryChunkedBuilder::new(name, len);

        for res in values.iter()? {
            let value = res?;
            if value.is_none() {
                builder.append_null()
            } else {
                let v = value.extract::<&[u8]>()?;
                builder.append_value(v)
            }
        }

        let ca = builder.finish();
        let s = ca.into_series();
        Ok(s.into())
    }

    #[staticmethod]
    fn new_decimal(name: &str, values: &Bound<PyAny>, strict: bool) -> PyResult<Self> {
        Self::new_from_any_values(name, values, strict)
    }

    #[staticmethod]
    fn new_series_list(name: &str, values: Vec<Option<PySeries>>, _strict: bool) -> PyResult<Self> {
        let series = reinterpret_vec(values);
        if let Some(s) = series.iter().flatten().next() {
            if s.dtype().is_object() {
                return Err(PyValueError::new_err(
                    "list of objects isn't supported; try building a 'object' only series",
                ));
            }
        }
        Ok(Series::new(name, series).into())
    }

    #[staticmethod]
    #[pyo3(signature = (name, values, strict, dtype))]
    fn new_array(
        name: &str,
        values: &Bound<PyAny>,
        strict: bool,
        dtype: Wrap<DataType>,
    ) -> PyResult<Self> {
        Self::new_from_any_values_and_dtype(name, values, dtype, strict)
    }

    #[staticmethod]
    pub fn new_object(py: Python, name: &str, values: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            let mut validity = MutableBitmap::with_capacity(values.len());
            values.iter().for_each(|v| {
                let is_valid = !v.inner.is_none(py);
                // SAFETY: we can ensure that validity has correct capacity.
                unsafe { validity.push_unchecked(is_valid) };
            });
            // Object builder must be registered. This is done on import.
            let ca = ObjectChunked::<ObjectValue>::new_from_vec_and_validity(
                name,
                values,
                validity.into(),
            );
            let s = ca.into_series();
            s.into()
        }
        #[cfg(not(feature = "object"))]
        panic!("activate 'object' feature")
    }

    #[staticmethod]
    fn new_null(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        Ok(Series::new_null(name, len).into())
    }

    #[staticmethod]
    fn from_arrow(name: &str, array: &Bound<PyAny>) -> PyResult<Self> {
        let arr = array_to_rust(array)?;

        match arr.data_type() {
            ArrowDataType::LargeList(_) => {
                let array = arr.as_any().downcast_ref::<LargeListArray>().unwrap();
                let fast_explode = array.offsets().as_slice().windows(2).all(|w| w[0] != w[1]);

                let mut out = ListChunked::with_chunk(name, array.clone());
                if fast_explode {
                    out.set_fast_explode()
                }
                Ok(out.into_series().into())
            },
            _ => {
                let series: Series =
                    std::convert::TryFrom::try_from((name, arr)).map_err(PyPolarsErr::from)?;
                Ok(series.into())
            },
        }
    }
}
