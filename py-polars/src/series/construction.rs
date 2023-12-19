use numpy::{Element, PyArray1};
use polars::export::arrow;
use polars::export::arrow::array::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use polars::export::arrow::bitmap::Bitmap;
use polars::export::arrow::buffer::Buffer;
use polars::export::arrow::offset::OffsetsBuffer;
use polars::export::arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::utils::CustomIterTools;
use pyo3::exceptions::{PyTypeError, PyValueError};
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
    let ro_array = array.readonly();
    let vals = ro_array.as_slice().unwrap();

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
    pub fn new_object(name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            // Object builder must be registered. This is done on import.
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

    #[staticmethod]
    unsafe fn _from_buffers(
        dtype: Wrap<DataType>,
        data: PySeries,
        validity: Option<PySeries>,
        offsets: Option<PySeries>,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let data = data.series;

        let validity = match validity {
            Some(s) => Some(series_to_bitmap(s.series)?),
            None => None,
        };

        let s = match dtype {
            DataType::Int8 => {
                let data = series_to_buffer::<Int8Type>(data)?;
                from_buffers_num_impl::<i8>(data, validity)?
            },
            DataType::Int16 => {
                let data = series_to_buffer::<Int16Type>(data)?;
                from_buffers_num_impl::<i16>(data, validity)?
            },
            DataType::Int32 => {
                let data = series_to_buffer::<Int32Type>(data)?;
                from_buffers_num_impl::<i32>(data, validity)?
            },
            DataType::Int64 => {
                let data = series_to_buffer::<Int64Type>(data)?;
                from_buffers_num_impl::<i64>(data, validity)?
            },
            DataType::UInt8 => {
                let data = series_to_buffer::<UInt8Type>(data)?;
                from_buffers_num_impl::<u8>(data, validity)?
            },
            DataType::UInt16 => {
                let data = series_to_buffer::<UInt16Type>(data)?;
                from_buffers_num_impl::<u16>(data, validity)?
            },
            DataType::UInt32 => {
                let data = series_to_buffer::<UInt32Type>(data)?;
                from_buffers_num_impl::<u32>(data, validity)?
            },
            DataType::UInt64 => {
                let data = series_to_buffer::<UInt64Type>(data)?;
                from_buffers_num_impl::<u64>(data, validity)?
            },
            DataType::Float32 => {
                let data = series_to_buffer::<Float32Type>(data)?;
                from_buffers_num_impl::<f32>(data, validity)?
            },
            DataType::Float64 => {
                let data = series_to_buffer::<Float64Type>(data)?;
                from_buffers_num_impl::<f64>(data, validity)?
            },
            DataType::Boolean => {
                let data = series_to_bitmap(data)?;
                from_buffers_bool_impl(data, validity)?
            },
            DataType::Utf8 => {
                let data = series_to_buffer::<UInt8Type>(data)?;
                let offsets =
                    match offsets {
                        Some(s) => series_to_offsets(s.series)?,
                        None => return Err(PyTypeError::new_err(
                            "`from_buffers` cannot create a Utf8 column without an offsets buffer",
                        )),
                    };
                from_buffers_string_impl(data, validity, offsets)?
            },
            DataType::Date => {
                let data = series_to_buffer::<Int32Type>(data)?;
                let physical = from_buffers_num_impl::<i32>(data, validity)?;
                physical.cast(&DataType::Date).map_err(PyPolarsErr::from)?
            },
            DataType::Time => {
                let data = series_to_buffer::<Int64Type>(data)?;
                let physical = from_buffers_num_impl::<i64>(data, validity)?;
                physical.cast(&DataType::Time).map_err(PyPolarsErr::from)?
            },
            DataType::Datetime(tu, tz) => {
                let data = series_to_buffer::<Int64Type>(data)?;
                let physical = from_buffers_num_impl::<i64>(data, validity)?;
                physical
                    .cast(&DataType::Datetime(tu, tz))
                    .map_err(PyPolarsErr::from)?
            },
            DataType::Duration(tu) => {
                let data = series_to_buffer::<Int64Type>(data)?;
                let physical = from_buffers_num_impl::<i64>(data, validity)?;
                physical
                    .cast(&DataType::Duration(tu))
                    .map_err(PyPolarsErr::from)?
            },
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "`from_buffers` not implemented for `dtype` {dt}",
                )))
            },
        };

        Ok(s.into())
    }

    #[staticmethod]
    unsafe fn _from_buffer(
        py: Python,
        pointer: usize,
        offset: usize,
        length: usize,
        dtype: Wrap<DataType>,
        base: &PyAny,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let base = base.to_object(py);

        let s = match dtype {
            DataType::Int8 => unsafe { from_buffer_impl::<i8>(pointer, length, base) },
            DataType::Int16 => unsafe { from_buffer_impl::<i16>(pointer, length, base) },
            DataType::Int32 => unsafe { from_buffer_impl::<i32>(pointer, length, base) },
            DataType::Int64 => unsafe { from_buffer_impl::<i64>(pointer, length, base) },
            DataType::UInt8 => unsafe { from_buffer_impl::<u8>(pointer, length, base) },
            DataType::UInt16 => unsafe { from_buffer_impl::<u16>(pointer, length, base) },
            DataType::UInt32 => unsafe { from_buffer_impl::<u32>(pointer, length, base) },
            DataType::UInt64 => unsafe { from_buffer_impl::<u64>(pointer, length, base) },
            DataType::Float32 => unsafe { from_buffer_impl::<f32>(pointer, length, base) },
            DataType::Float64 => unsafe { from_buffer_impl::<f64>(pointer, length, base) },
            DataType::Boolean => {
                unsafe { from_buffer_boolean_impl(pointer, offset, length, base) }?
            },
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "`from_buffer` requires a physical type as input for `dtype`, got {dt}",
                )))
            },
        };
        Ok(s.into())
    }
}

unsafe fn from_buffer_impl<T: NativeType>(
    pointer: usize,
    length: usize,
    base: Py<PyAny>,
) -> Series {
    let pointer = pointer as *const T;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length) };
    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(slice, base) };
    Series::from_arrow("", arr.to_boxed()).unwrap()
}

unsafe fn from_buffer_boolean_impl(
    pointer: usize,
    offset: usize,
    length: usize,
    base: Py<PyAny>,
) -> PyResult<Series> {
    let length_in_bytes = get_boolean_buffer_length_in_bytes(length, offset);

    let pointer = pointer as *const u8;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length_in_bytes) };
    let arr_result = unsafe { arrow::ffi::mmap::bitmap_and_owner(slice, offset, length, base) };
    let arr = arr_result.map_err(PyPolarsErr::from)?;
    let s = Series::from_arrow("", arr.to_boxed()).unwrap();
    Ok(s)
}
fn get_boolean_buffer_length_in_bytes(length: usize, offset: usize) -> usize {
    let n_bits = offset + length;
    let n_bytes = n_bits / 8;
    let rest = n_bits % 8;
    if rest == 0 {
        n_bytes
    } else {
        n_bytes + 1
    }
}

fn series_to_buffer<T>(s: Series) -> PyResult<Buffer<T::Native>>
where
    T: PolarsNumericType,
{
    let ca: &ChunkedArray<T> = s.as_any().downcast_ref().unwrap();
    let arr = ca.downcast_iter().next().unwrap();
    let values = arr.values().clone();
    Ok(values)
}
fn series_to_bitmap(s: Series) -> PyResult<Bitmap> {
    let ca_result = s.bool();
    let ca = ca_result.map_err(PyPolarsErr::from)?;
    let arr = ca.downcast_iter().next().unwrap();
    let bitmap = arr.values().clone();
    Ok(bitmap)
}
fn series_to_offsets(s: Series) -> PyResult<OffsetsBuffer<i64>> {
    let ca_result = s.i64();
    let ca = ca_result.map_err(PyPolarsErr::from)?;
    let arr = ca.downcast_iter().next().unwrap();
    let buffer = arr.values().clone();
    let offsets = unsafe { OffsetsBuffer::new_unchecked(buffer) };
    Ok(offsets)
}

fn from_buffers_num_impl<T: NativeType>(
    data: Buffer<T>,
    validity: Option<Bitmap>,
) -> PyResult<Series> {
    let arr = PrimitiveArray::new(T::PRIMITIVE.into(), data, validity);
    let s_result = Series::from_arrow("", arr.to_boxed());
    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
fn from_buffers_bool_impl(data: Bitmap, validity: Option<Bitmap>) -> PyResult<Series> {
    let arr = BooleanArray::new(ArrowDataType::Boolean, data, validity);
    let s_result = Series::from_arrow("", arr.to_boxed());
    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
fn from_buffers_string_impl(
    data: Buffer<u8>,
    validity: Option<Bitmap>,
    offsets: OffsetsBuffer<i64>,
) -> PyResult<Series> {
    let arr = Utf8Array::new(ArrowDataType::LargeUtf8, offsets, data, validity);
    let s_result = Series::from_arrow("", arr.to_boxed());
    let s = s_result.map_err(PyPolarsErr::from)?;
    Ok(s)
}
