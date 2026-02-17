use std::borrow::Cow;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::BitmapBuilder;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use numpy::{Element, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use polars::prelude::*;
use polars_buffer::{Buffer, SharedStorage};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::PySeries;
use crate::conversion::Wrap;
use crate::conversion::any_value::py_object_to_any_value;
use crate::error::PyPolarsErr;
use crate::interop::arrow::to_rust::array_to_rust;
use crate::prelude::ObjectValue;
use crate::utils::EnterPolarsExt;

// Init with numpy arrays.
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(name: &str, array: &Bound<PyArray1<$type>>, _strict: bool) -> Self {
                let arr = numpy_array_to_arrow(array);
                Series::from_arrow(name.into(), arr.to_boxed())
                    .unwrap()
                    .into()
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

fn numpy_array_to_arrow<T: Element + NativeType>(array: &Bound<PyArray1<T>>) -> PrimitiveArray<T> {
    let owner = array.clone().unbind();
    let ro = array.readonly();
    let vals = ro.as_slice().unwrap();
    unsafe {
        let storage = SharedStorage::from_slice_with_owner(vals, owner);
        let buffer = Buffer::from_storage(storage);
        PrimitiveArray::new_unchecked(T::PRIMITIVE.into(), buffer, None)
    }
}

#[cfg(feature = "object")]
pub fn series_from_objects(py: Python<'_>, name: PlSmallStr, objects: Vec<ObjectValue>) -> Series {
    let mut validity = BitmapBuilder::with_capacity(objects.len());
    for v in &objects {
        let is_valid = !v.inner.is_none(py);
        // SAFETY: we can ensure that validity has correct capacity.
        unsafe { validity.push_unchecked(is_valid) };
    }
    ObjectChunked::<ObjectValue>::new_from_vec_and_validity(
        name,
        objects,
        validity.into_opt_validity(),
    )
    .into_series()
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_bool(
        py: Python<'_>,
        name: &str,
        array: &Bound<PyArray1<bool>>,
        _strict: bool,
    ) -> PyResult<Self> {
        let array = array.readonly();

        // We use raw ptr methods to read this as a u8 slice to work around PyO3/rust-numpy#509.
        assert!(array.is_contiguous());
        let data_ptr = array.data().cast::<u8>();
        let data_len = array.len();
        let vals = unsafe { core::slice::from_raw_parts(data_ptr, data_len) };
        py.enter_polars_series(|| Series::new(name.into(), vals).cast(&DataType::Boolean))
    }

    #[staticmethod]
    fn new_f16(
        py: Python<'_>,
        name: &str,
        array: &Bound<PyArray1<pf16>>,
        nan_is_null: bool,
    ) -> PyResult<Self> {
        let arr = numpy_array_to_arrow(array);
        if nan_is_null {
            py.enter_polars_series(|| {
                let validity = polars_compute::nan::is_not_nan(arr.values());
                Ok(Series::from_array(name.into(), arr.with_validity(validity)))
            })
        } else {
            Ok(Series::from_array(name.into(), arr).into())
        }
    }

    #[staticmethod]
    fn new_f32(
        py: Python<'_>,
        name: &str,
        array: &Bound<PyArray1<f32>>,
        nan_is_null: bool,
    ) -> PyResult<Self> {
        let arr = numpy_array_to_arrow(array);
        if nan_is_null {
            py.enter_polars_series(|| {
                let validity = polars_compute::nan::is_not_nan(arr.values());
                Ok(Series::from_array(name.into(), arr.with_validity(validity)))
            })
        } else {
            Ok(Series::from_array(name.into(), arr).into())
        }
    }

    #[staticmethod]
    fn new_f64(
        py: Python<'_>,
        name: &str,
        array: &Bound<PyArray1<f64>>,
        nan_is_null: bool,
    ) -> PyResult<Self> {
        let arr = numpy_array_to_arrow(array);
        if nan_is_null {
            py.enter_polars_series(|| {
                let validity = polars_compute::nan::is_not_nan(arr.values());
                Ok(Series::from_array(name.into(), arr.with_validity(validity)))
            })
        } else {
            Ok(Series::from_array(name.into(), arr).into())
        }
    }
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_opt_bool(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        let mut builder = BooleanChunkedBuilder::new(name.into(), len);

        for res in values.try_iter()? {
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

fn new_primitive<'py, T, F>(
    name: &str,
    values: &Bound<'py, PyAny>,
    _strict: bool,
    extract: F,
) -> PyResult<PySeries>
where
    T: PolarsNumericType,
    F: Fn(Bound<'py, PyAny>) -> PyResult<T::Native>,
{
    let len = values.len()?;
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name.into(), len);

    for res in values.try_iter()? {
        let value = res?;
        if value.is_none() {
            builder.append_null()
        } else {
            let v = extract(value)?;
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
                new_primitive::<$type, _>(name, obj, strict, |v| v.extract::<$native>())
            }
        }
    };
}

init_method_opt!(new_opt_u8, UInt8Type, u8);
init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_u128, UInt128Type, u128);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_i128, Int128Type, i128);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_opt_f16(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        new_primitive::<Float16Type, _>(name, values, false, |v| {
            Ok(AsPrimitive::<pf16>::as_(v.extract::<f64>()?))
        })
    }
}

fn convert_to_avs(
    values: &Bound<'_, PyAny>,
    strict: bool,
    allow_object: bool,
) -> PyResult<Vec<AnyValue<'static>>> {
    values
        .try_iter()?
        .map(|v| py_object_to_any_value(&(v?).as_borrowed(), strict, allow_object))
        .collect()
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_from_any_values(name: &str, values: &Bound<PyAny>, strict: bool) -> PyResult<Self> {
        let any_values_result = values
            .try_iter()?
            .map(|v| py_object_to_any_value(&(v?).as_borrowed(), strict, true))
            .collect::<PyResult<Vec<AnyValue>>>();

        let result = any_values_result.and_then(|avs| {
            let s = Series::from_any_values(name.into(), avs.as_slice(), strict).map_err(|e| {
                PyTypeError::new_err(format!(
                    "{e}\n\nHint: Try setting `strict=False` to allow passing data with mixed types."
                ))
            })?;
            Ok(s.into())
        });

        // Fall back to Object type for non-strict construction.
        if !strict && result.is_err() {
            return Python::attach(|py| {
                let objects = values
                    .try_iter()?
                    .map(|v| v?.extract())
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
        let avs = convert_to_avs(values, strict, false)?;
        let s = Series::from_any_values_and_dtype(name.into(), avs.as_slice(), &dtype.0, strict)
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
        let mut builder = StringChunkedBuilder::new(name.into(), len);

        for res in values.try_iter()? {
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
        let mut builder = BinaryChunkedBuilder::new(name.into(), len);

        for res in values.try_iter()? {
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
        let series: Vec<_> = values
            .into_iter()
            .map(|ops| ops.map(|ps| ps.series.into_inner()))
            .collect();
        if let Some(s) = series.iter().flatten().next() {
            if s.dtype().is_object() {
                return Err(PyValueError::new_err(
                    "list of objects isn't supported; try building a 'object' only series",
                ));
            }
        }
        Ok(Series::new(name.into(), series).into())
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
    pub fn new_object(py: Python<'_>, name: &str, values: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            PySeries::from(series_from_objects(py, name.into(), values))
        }
        #[cfg(not(feature = "object"))]
        panic!("activate 'object' feature")
    }

    #[staticmethod]
    fn new_null(name: &str, values: &Bound<PyAny>, _strict: bool) -> PyResult<Self> {
        let len = values.len()?;
        Ok(Series::new_null(name.into(), len).into())
    }

    #[staticmethod]
    fn from_arrow(name: &str, array: &Bound<PyAny>) -> PyResult<Self> {
        let arr = array_to_rust(array)?;

        match arr.dtype() {
            ArrowDataType::LargeList(_) => {
                let array = arr.as_any().downcast_ref::<LargeListArray>().unwrap();
                let fast_explode = array.offsets().as_slice().windows(2).all(|w| w[0] != w[1]);

                let mut out = ListChunked::with_chunk(name.into(), array.clone());
                if fast_explode {
                    out.set_fast_explode()
                }
                Ok(out.into_series().into())
            },
            _ => {
                let series: Series =
                    Series::try_new(name.into(), arr).map_err(PyPolarsErr::from)?;
                Ok(series.into())
            },
        }
    }
}
