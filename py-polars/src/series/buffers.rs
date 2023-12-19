use polars::export::arrow;
use polars::export::arrow::array::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use polars::export::arrow::bitmap::Bitmap;
use polars::export::arrow::buffer::Buffer;
use polars::export::arrow::offset::OffsetsBuffer;
use polars::export::arrow::types::NativeType;
use polars_core::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::*;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::PySeries;

#[pymethods]
impl PySeries {
    /// Returns `(offset, len, ptr)`
    fn get_ptr(&self) -> PyResult<(usize, usize, usize)> {
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
                let (slice, start, len) = arr.values().as_slice();
                Ok((start, len, slice.as_ptr() as usize))
            },
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                (0, ca.len(), get_ptr(ca))
            })),
            DataType::String => {
                let ca = s.str().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                Ok((0, arr.len(), arr.values().as_ptr() as usize))
            },
            DataType::Binary => {
                let ca = s.binary().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                Ok((0, arr.len(), arr.values().as_ptr() as usize))
            },
            _ => {
                let msg = "Cannot take pointer of nested type, try to first select a buffer";
                raise_err!(msg, ComputeError);
            },
        }
    }

    fn get_buffer(&self, index: usize) -> PyResult<Option<Self>> {
        match self.series.dtype().to_physical() {
            dt if dt.is_numeric() => get_buffer_from_primitive(&self.series, index),
            DataType::Boolean => get_buffer_from_primitive(&self.series, index),
            DataType::String | DataType::List(_) | DataType::Binary => {
                get_buffer_from_nested(&self.series, index)
            },
            DataType::Array(_, _) => {
                let ca = self.series.array().unwrap();
                match index {
                    0 => {
                        let buffers = ca
                            .downcast_iter()
                            .map(|arr| arr.values().clone())
                            .collect::<Vec<_>>();
                        Ok(Some(
                            Series::try_from((self.series.name(), buffers))
                                .map_err(PyPolarsErr::from)?
                                .into(),
                        ))
                    },
                    1 => Ok(get_bitmap(&self.series)),
                    2 => Ok(None),
                    _ => Err(PyValueError::new_err("expected an index <= 2")),
                }
            },
            _ => todo!(),
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

fn get_bitmap(s: &Series) -> Option<PySeries> {
    if s.null_count() > 0 {
        Some(s.is_not_null().into_series().into())
    } else {
        None
    }
}

fn get_buffer_from_nested(s: &Series, index: usize) -> PyResult<Option<PySeries>> {
    match index {
        0 => {
            let buffers: Box<dyn Iterator<Item = ArrayRef>> = match s.dtype() {
                DataType::List(_) => {
                    let ca = s.list().unwrap();
                    Box::new(ca.downcast_iter().map(|arr| arr.values().clone()))
                },
                DataType::String => {
                    let ca = s.str().unwrap();
                    Box::new(ca.downcast_iter().map(|arr| {
                        PrimitiveArray::from_data_default(arr.values().clone(), None).boxed()
                    }))
                },
                DataType::Binary => {
                    let ca = s.binary().unwrap();
                    Box::new(ca.downcast_iter().map(|arr| {
                        PrimitiveArray::from_data_default(arr.values().clone(), None).boxed()
                    }))
                },
                dt => {
                    let msg = format!("{dt} not yet supported as nested buffer access");
                    raise_err!(msg, ComputeError);
                },
            };
            let buffers = buffers.collect::<Vec<_>>();
            Ok(Some(
                Series::try_from((s.name(), buffers))
                    .map_err(PyPolarsErr::from)?
                    .into(),
            ))
        },
        1 => Ok(get_bitmap(s)),
        2 => get_offsets(s).map(Some),
        _ => Err(PyValueError::new_err("expected an index <= 2")),
    }
}

fn get_offsets(s: &Series) -> PyResult<PySeries> {
    let buffers: Box<dyn Iterator<Item = &OffsetsBuffer<i64>>> = match s.dtype() {
        DataType::List(_) => {
            let ca = s.list().unwrap();
            Box::new(ca.downcast_iter().map(|arr| arr.offsets()))
        },
        DataType::String => {
            let ca = s.str().unwrap();
            Box::new(ca.downcast_iter().map(|arr| arr.offsets()))
        },
        _ => return Err(PyValueError::new_err("expected list/string")),
    };
    let buffers = buffers
        .map(|arr| PrimitiveArray::from_data_default(arr.buffer().clone(), None).boxed())
        .collect::<Vec<_>>();
    Ok(Series::try_from((s.name(), buffers))
        .map_err(PyPolarsErr::from)?
        .into())
}

fn get_buffer_from_primitive(s: &Series, index: usize) -> PyResult<Option<PySeries>> {
    match index {
        0 => {
            let chunks = s
                .chunks()
                .iter()
                .map(|arr| arr.with_validity(None))
                .collect::<Vec<_>>();
            Ok(Some(
                Series::try_from((s.name(), chunks))
                    .map_err(PyPolarsErr::from)?
                    .into(),
            ))
        },
        1 => Ok(get_bitmap(s)),
        2 => Ok(None),
        _ => Err(PyValueError::new_err("expected an index <= 2")),
    }
}

fn get_ptr<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
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
