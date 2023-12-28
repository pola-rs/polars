use polars::export::arrow;
use polars::export::arrow::array::Array;
use polars::export::arrow::types::NativeType;
use polars_core::export::arrow::array::PrimitiveArray;
use polars_rs::export::arrow::offset::OffsetsBuffer;
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
                // this one is quite useless as you need to know the offset
                // into the first byte.
                let (slice, start, len) = arr.values().as_slice();
                Ok(BufferInfo {
                    pointer: slice.as_ptr() as usize,
                    offset: start,
                    length: len,
                })
            },
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                BufferInfo { pointer: get_pointer(ca), offset: 0, length: ca.len() }
            })),
            DataType::String => {
                let ca = s.str().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                Ok(BufferInfo {
                    pointer: arr.values().as_ptr() as usize,
                    offset: 0,
                    length: arr.len(),
                })
            },
            DataType::Binary => {
                let ca = s.binary().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                Ok(BufferInfo {
                    pointer: arr.values().as_ptr() as usize,
                    offset: 0,
                    length: arr.len(),
                })
            },
            _ => {
                let msg = "Cannot take pointer of nested type, try to first select a buffer";
                raise_err!(msg, ComputeError);
            },
        }
    }

    /// Return the underlying data, validity, or offsets buffer as a Series.
    fn _get_buffer(&self, index: usize) -> PyResult<Option<Self>> {
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
        pointer: usize,
        offset: usize,
        length: usize,
        base: &PyAny,
    ) -> PyResult<Self> {
        let dtype = dtype.0;
        let base = base.to_object(py);

        let arr_boxed = match dtype {
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

        let s = Series::from_arrow("", arr_boxed).unwrap().into();
        Ok(s)
    }
}

unsafe fn from_buffer_impl<T: NativeType>(
    pointer: usize,
    length: usize,
    base: Py<PyAny>,
) -> Box<dyn Array> {
    let pointer = pointer as *const T;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length) };
    let arr = unsafe { arrow::ffi::mmap::slice_and_owner(slice, base) };
    arr.to_boxed()
}
unsafe fn from_buffer_boolean_impl(
    pointer: usize,
    offset: usize,
    length: usize,
    base: Py<PyAny>,
) -> PyResult<Box<dyn Array>> {
    let length_in_bytes = get_boolean_buffer_length_in_bytes(length, offset);

    let pointer = pointer as *const u8;
    let slice = unsafe { std::slice::from_raw_parts(pointer, length_in_bytes) };
    let arr_result = unsafe { arrow::ffi::mmap::bitmap_and_owner(slice, offset, length, base) };
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
