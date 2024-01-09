use numpy::{Element, PyArray1};
use polars::export::arrow;
use polars::export::arrow::array::Array;
use polars::export::arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::utils::CustomIterTools;
use polars_rs::export::arrow::bitmap::MutableBitmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::arrow_interop::to_rust::array_to_rust;
use crate::conversion::{slice_extract_wrapped, vec_extract_wrapped, Wrap};
use crate::error::PyPolarsErr;
use crate::prelude::ObjectValue;
use crate::series::ToSeries;
use crate::PySeries;

// Init with numpy arrays.
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(py: Python, name: &str, array: &PyArray1<$type>, _strict: bool) -> PySeries {
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
    array: &PyArray1<T>,
) -> PySeries {
    let vals = unsafe { array.as_slice().unwrap() };

    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(vals, array.to_object(py)) };
    Series::from_arrow(name, arr.to_boxed()).unwrap().into()
}

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_bool(py: Python, name: &str, array: &PyArray1<bool>, _strict: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| PySeries {
            series: Series::new(name, vals),
        })
    }

    #[staticmethod]
    fn new_f32(py: Python, name: &str, array: &PyArray1<f32>, nan_is_null: bool) -> PySeries {
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
    fn new_f64(py: Python, name: &str, array: &PyArray1<f64>, nan_is_null: bool) -> PySeries {
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
    fn new_opt_bool(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
        let len = obj.len()?;
        let mut builder = BooleanChunkedBuilder::new(name, len);

        for res in obj.iter()? {
            let item = res?;
            if item.is_none() {
                builder.append_null()
            } else {
                match item.extract::<bool>() {
                    Ok(val) => builder.append_value(val),
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        builder.append_null()
                    },
                }
            }
        }
        let ca = builder.finish();

        let s = ca.into_series();
        Ok(PySeries { series: s })
    }
}

fn new_primitive<'a, T>(name: &str, obj: &'a PyAny, strict: bool) -> PyResult<PySeries>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    T::Native: FromPyObject<'a>,
{
    let len = obj.len()?;
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name, len);

    for res in obj.iter()? {
        let item = res?;

        if item.is_none() {
            builder.append_null()
        } else {
            match item.extract::<T::Native>() {
                Ok(val) => builder.append_value(val),
                Err(e) => {
                    if strict {
                        return Err(e);
                    }
                    builder.append_null()
                },
            }
        }
    }
    let ca = builder.finish();

    let s = ca.into_series();
    Ok(PySeries { series: s })
}

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
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
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PySeries {
    #[staticmethod]
    fn new_from_anyvalues(
        name: &str,
        val: Vec<Wrap<AnyValue<'_>>>,
        strict: bool,
    ) -> PyResult<PySeries> {
        // From AnyValues is fallible.
        let avs = slice_extract_wrapped(&val);
        let s = Series::from_any_values(name, avs, strict).map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn new_from_anyvalues_and_dtype(
        name: &str,
        val: Vec<Wrap<AnyValue<'_>>>,
        dtype: Wrap<DataType>,
        strict: bool,
    ) -> PyResult<PySeries> {
        let avs = slice_extract_wrapped(&val);
        let s = Series::from_any_values_and_dtype(name, avs, &dtype.0, strict)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn new_str(name: &str, val: Wrap<StringChunked>, _strict: bool) -> Self {
        val.0.into_series().with_name(name).into()
    }

    #[staticmethod]
    fn new_binary(name: &str, val: Wrap<BinaryChunked>, _strict: bool) -> Self {
        val.0.into_series().with_name(name).into()
    }

    #[staticmethod]
    fn new_null(name: &str, val: &PyAny, _strict: bool) -> PyResult<Self> {
        Ok(Series::new_null(name, val.len()?).into())
    }

    #[staticmethod]
    pub fn new_object(py: Python, name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            let mut validity = MutableBitmap::with_capacity(val.len());
            val.iter().for_each(|v| {
                if v.inner.is_none(py) {
                    // SAFETY: we can ensure that validity has correct capacity.
                    unsafe { validity.push_unchecked(false) };
                } else {
                    // SAFETY: we can ensure that validity has correct capacity.
                    unsafe { validity.push_unchecked(true) };
                }
            });
            // Object builder must be registered. This is done on import.
            let s =
                ObjectChunked::<ObjectValue>::new_from_vec_and_validity(name, val, validity.into())
                    .into_series();
            s.into()
        }
        #[cfg(not(feature = "object"))]
        {
            todo!()
        }
    }

    #[staticmethod]
    fn new_series_list(name: &str, val: Vec<PySeries>, _strict: bool) -> Self {
        let series_vec = val.to_series();
        Series::new(name, &series_vec).into()
    }

    #[staticmethod]
    #[pyo3(signature = (width, inner, name, val, _strict))]
    fn new_array(
        width: usize,
        inner: Option<Wrap<DataType>>,
        name: &str,
        val: Vec<Wrap<AnyValue>>,
        _strict: bool,
    ) -> PyResult<Self> {
        if val.is_empty() {
            let series =
                Series::new_empty(name, &DataType::Array(Box::new(inner.unwrap().0), width));
            Ok(series.into())
        } else {
            let val = vec_extract_wrapped(val);
            return if let Some(inner) = inner {
                let series = Series::from_any_values_and_dtype(
                    name,
                    val.as_ref(),
                    &DataType::Array(Box::new(inner.0), width),
                    true,
                )
                .map_err(PyPolarsErr::from)?;
                Ok(series.into())
            } else {
                let series = Series::new(name, &val);
                match series.dtype() {
                    DataType::List(list_inner) => {
                        let series = series
                            .cast(&DataType::Array(
                                Box::new(inner.map(|dt| dt.0).unwrap_or(*list_inner.clone())),
                                width,
                            ))
                            .map_err(PyPolarsErr::from)?;
                        Ok(series.into())
                    },
                    _ => Err(PyValueError::new_err("could not create Array from input")),
                }
            };
        }
    }

    #[staticmethod]
    fn new_decimal(name: &str, val: Vec<Wrap<AnyValue<'_>>>, strict: bool) -> PyResult<PySeries> {
        // TODO: do we have to respect 'strict' here? It's possible if we want to.
        let avs = slice_extract_wrapped(&val);
        // Create a fake dtype with a placeholder "none" scale, to be inferred later.
        let dtype = DataType::Decimal(None, None);
        let s = Series::from_any_values_and_dtype(name, avs, &dtype, strict)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn from_arrow(name: &str, array: &PyAny) -> PyResult<Self> {
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
