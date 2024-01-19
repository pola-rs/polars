use polars::export::arrow;
use polars::export::arrow::array::{Array, BooleanArray, PrimitiveArray, Utf8Array};
use polars::export::arrow::bitmap::Bitmap;
use polars::export::arrow::buffer::Buffer;
use polars::export::arrow::offset::OffsetsBuffer;
use polars::export::arrow::types::NativeType;
use pyo3::exceptions::{PyTypeError, PyValueError};

use super::*;

struct BufferInfo {
    pointer: usize,
    offset: usize,
    length: usize,
}
impl IntoPy<PyObject> for BufferInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (self.pointer, self.offset, self.length).to_object(py)
    }
}
impl<'a> FromPyObject<'a> for BufferInfo {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let (pointer, offset, length) = ob.extract()?;
        Ok(Self {
            pointer,
            offset,
            length,
        })
    }
}

#[pymethods]
impl PySeries {
    /// Return pointer, offset, and length information about the underlying buffer.
    fn _get_buffer_info(&self) -> PyResult<BufferInfo> {
        let s = self.series.to_physical_repr();
        let arrays = s.chunks();
        if arrays.len() != 1 {
            let msg = "cannot get buffer info for Series consisting of multiple chunks";
            raise_err!(msg, ComputeError);
        }
        match s.dtype() {
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                let (slice, offset, len) = arr.values().as_slice();
                Ok(BufferInfo {
                    pointer: slice.as_ptr() as usize,
                    offset: offset,
                    length: len,
                })
            },
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                BufferInfo { pointer: get_pointer(ca), offset: 0, length: ca.len() }
            })),
            _ => {
                let msg = "cannot take pointer non-physical type; try to select a buffer first";
                raise_err!(msg, ComputeError);
            },
        }
    }

    /// Return the underlying data, validity, or offsets buffer as a Series.
    fn _get_buffer(&self, index: usize) -> PyResult<Option<Self>> {
        match self.series.dtype().to_physical() {
            dt if dt.is_numeric() => get_buffer_from_primitive(&self.series, index),
            DataType::Boolean => get_buffer_from_primitive(&self.series, index),
            DataType::String => {
                todo!()
            },
            DataType::List(_) => get_buffer_from_nested(&self.series, index),
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
            let mut offsets = Vec::with_capacity(ca.len() + 1);
            let mut len_so_far: i64 = 0;
            offsets.push(len_so_far);

            for arr in ca.downcast_iter() {
                for length in arr.len_iter() {
                    len_so_far += length as i64;
                }
                offsets.push(len_so_far)
            }
            return Ok(Series::from_vec(s.name(), offsets).into());
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

fn get_pointer<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
}

#[pymethods]
impl PySeries {
    /// Construct a PySeries from information about its underlying buffer.
    #[staticmethod]
    unsafe fn _from_buffer(
        py: Python,
        dtype: Wrap<DataType>,
        buffer_info: BufferInfo,
        owner: &PyAny,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let BufferInfo {
            pointer,
            offset,
            length,
        } = buffer_info;
        let owner = owner.to_object(py);

        let arr_boxed = match dtype {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_type!(dt, |$T|  unsafe {
                    from_buffer_impl::<$T>(pointer, offset, length, owner)
                })
            },
            DataType::Boolean => {
                unsafe { from_buffer_boolean_impl(pointer, offset, length, owner) }?
            },
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "`from_buffer` requires a physical type as input for `dtype`, got {dt}",
                )))
            },
        };

        let s = Series::from_arrow("", arr_boxed).unwrap().into();
        Ok(s)
    }
}

unsafe fn from_buffer_impl<T: NativeType>(
    pointer: usize,
    offset: usize,
    length: usize,
    owner: Py<PyAny>,
) -> Box<dyn Array> {
    let pointer = pointer as *const T;
    let pointer = unsafe { pointer.add(offset) };
    let slice = unsafe { std::slice::from_raw_parts(pointer, length) };
    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(slice, owner) };
    arr.to_boxed()
}
unsafe fn from_buffer_boolean_impl(
    pointer: usize,
    offset: usize,
    length: usize,
    owner: Py<PyAny>,
) -> PyResult<Box<dyn Array>> {
    let length_in_bytes = get_boolean_buffer_length_in_bytes(length, offset);

    let pointer = pointer as *const u8;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length_in_bytes) };
    let arr_result = unsafe { arrow::ffi::mmap::bitmap_and_owner(slice, offset, length, owner) };
    let arr = arr_result.map_err(PyPolarsErr::from)?;
    Ok(arr.to_boxed())
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

#[pymethods]
impl PySeries {
    /// Construct a PySeries from information about its underlying buffers.
    #[staticmethod]
    unsafe fn _from_buffers(
        dtype: Wrap<DataType>,
        data: Vec<PySeries>,
        validity: Option<PySeries>,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let mut data = data.to_series();

        match data.len() {
            0 => {
                return Err(PyTypeError::new_err(
                    "`data` input to `from_buffers` must contain at least one buffer",
                ));
            },
            1 if validity.is_none() => {
                let values = data.pop().unwrap();
                let s = values.strict_cast(&dtype).map_err(PyPolarsErr::from)?;
                return Ok(s.into());
            },
            _ => (),
        }

        let validity = match validity {
            Some(s) => {
                let dtype = s.series.dtype();
                if !dtype.is_bool() {
                    return Err(PyTypeError::new_err(format!(
                        "validity buffer must have data type Boolean, got {:?}",
                        dtype
                    )));
                }
                Some(series_to_bitmap(s.series).unwrap())
            },
            None => None,
        };

        let s = match dtype.to_physical() {
            dt if dt.is_numeric() => {
                let values = data.into_iter().next().unwrap();
                with_match_physical_numeric_polars_type!(dt, |$T| {
                    let values_buffer = series_to_buffer::<$T>(values);
                    from_buffers_num_impl::<<$T as PolarsNumericType>::Native>(values_buffer, validity)?
                })
            },
            DataType::Boolean => {
                let values = data.into_iter().next().unwrap();
                let values_buffer = series_to_bitmap(values)?;
                from_buffers_bool_impl(values_buffer, validity)?
            },
            DataType::String => {
                let mut data_iter = data.into_iter();
                let values = data_iter.next().unwrap();
                let offsets = match data_iter.next() {
                    Some(s) => {
                        let dtype = s.dtype();
                        if !matches!(dtype, DataType::Int64) {
                            return Err(PyTypeError::new_err(format!(
                                "offsets buffer must have data type Int64, got {:?}",
                                dtype
                            )));
                        }
                        series_to_offsets(s)
                    },
                    None => return Err(PyTypeError::new_err(
                        "`from_buffers` cannot create a String column without an offsets buffer",
                    )),
                };
                let values = series_to_buffer::<UInt8Type>(values);
                from_buffers_string_impl(values, validity, offsets)?
            },
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "`from_buffers` not implemented for `dtype` {dt}",
                )))
            },
        };

        let out = s.strict_cast(&dtype).map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}

fn series_to_buffer<T>(s: Series) -> Buffer<T::Native>
where
    T: PolarsNumericType,
{
    let ca: &ChunkedArray<T> = s.as_ref().as_ref();
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().clone()
}
fn series_to_bitmap(s: Series) -> PyResult<Bitmap> {
    let ca_result = s.bool();
    let ca = ca_result.map_err(PyPolarsErr::from)?;
    let arr = ca.downcast_iter().next().unwrap();
    let bitmap = arr.values().clone();
    Ok(bitmap)
}
fn series_to_offsets(s: Series) -> OffsetsBuffer<i64> {
    let buffer = series_to_buffer::<Int64Type>(s);
    unsafe { OffsetsBuffer::new_unchecked(buffer) }
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
