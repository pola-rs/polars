use polars_core::export::arrow::array::PrimitiveArray;
use polars_rs::export::arrow::offset::OffsetsBuffer;

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
    /// Returns tuple with `(offset, len, ptr)`
    fn _get_buffer_info(&self) -> PyResult<BufferInfo> {
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
