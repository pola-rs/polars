use numpy::PyArray1;
use polars_core::prelude::*;
use polars_core::utils::CustomIterTools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::arrow_interop::to_rust::array_to_rust;
use crate::conversion::{slice_extract_wrapped, vec_extract_wrapped, Wrap};
use crate::error::PyPolarsErr;
use crate::prelude::ObjectValue;
use crate::series::ToSeries;
use crate::PySeries;

// Init with numpy arrays
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        #[pymethods]
        impl PySeries {
            #[staticmethod]
            fn $name(py: Python, name: &str, array: &PyArray1<$type>, _strict: bool) -> PySeries {
                let array = array.readonly();
                let vals = array.as_slice().unwrap();
                py.allow_threads(|| PySeries {
                    series: Series::new(name, vals),
                })
            }
        }
    };
}

init_method!(new_i8, i8);
init_method!(new_i16, i16);
init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_bool, bool);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);

#[pymethods]
impl PySeries {
    #[staticmethod]
    fn new_f32(py: Python, name: &str, array: &PyArray1<f32>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float32Chunked = vals
                    .iter()
                    .map(|&val| if f32::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
    }

    #[staticmethod]
    fn new_f64(py: Python, name: &str, array: &PyArray1<f64>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float64Chunked = vals
                    .iter()
                    .map(|&val| if f64::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
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
                    }
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
                }
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
        let avs = slice_extract_wrapped(&val);
        // from anyvalues is fallible
        let s = Series::from_any_values(name, avs, strict).map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn new_str(name: &str, val: Wrap<Utf8Chunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        s.into()
    }

    #[staticmethod]
    fn new_binary(name: &str, val: Wrap<BinaryChunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        s.into()
    }

    #[staticmethod]
    fn new_null(name: &str, val: &PyAny, _strict: bool) -> PyResult<Self> {
        let s = Series::new_null(name, val.len()?);
        Ok(s.into())
    }

    #[staticmethod]
    pub fn new_object(name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            // object builder must be registered. this is done on import
            let s = ObjectChunked::<ObjectValue>::new_from_vec(name, val).into_series();
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
        let val = vec_extract_wrapped(val);
        let out = Series::new(name, &val);
        match out.dtype() {
            DataType::List(list_inner) => {
                let out = out
                    .cast(&DataType::Array(
                        Box::new(inner.map(|dt| dt.0).unwrap_or(*list_inner.clone())),
                        width,
                    ))
                    .map_err(PyPolarsErr::from)?;
                Ok(out.into())
            }
            _ => Err(PyValueError::new_err("could not create Array from input")),
        }
    }

    #[staticmethod]
    fn new_decimal(name: &str, val: Vec<Wrap<AnyValue<'_>>>, strict: bool) -> PyResult<PySeries> {
        // TODO: do we have to respect 'strict' here? it's possible if we want to
        let avs = slice_extract_wrapped(&val);
        // create a fake dtype with a placeholder "none" scale, to be inferred later
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

                let mut previous = 0;
                let mut fast_explode = true;
                for &o in array.offsets().as_slice()[1..].iter() {
                    if o == previous {
                        fast_explode = false;
                        break;
                    }
                    previous = o;
                }
                let mut out = unsafe { ListChunked::from_chunks(name, vec![arr]) };
                if fast_explode {
                    out.set_fast_explode()
                }
                Ok(out.into_series().into())
            }
            _ => {
                let series: Series =
                    std::convert::TryFrom::try_from((name, arr)).map_err(PyPolarsErr::from)?;
                Ok(series.into())
            }
        }
    }
}
