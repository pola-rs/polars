use polars_core::export::arrow::array::PrimitiveArray;
use polars_rs::export::arrow::offset::OffsetsBuffer;

use super::*;

#[pymethods]
impl PySeries {
    fn offset_buffers(&self) -> PyResult<Self> {
        let buffers: Box<dyn Iterator<Item = &OffsetsBuffer<i64>>> = match self.series.dtype() {
            DataType::List(_) => {
                let ca = self.series.list().unwrap();
                Box::new(ca.downcast_iter().map(|arr| arr.offsets()))
            }
            DataType::Utf8 => {
                let ca = self.series.utf8().unwrap();
                Box::new(ca.downcast_iter().map(|arr| arr.offsets()))
            }
            _ => return Err(PyValueError::new_err("expected list/utf8")),
        };
        let buffers = buffers
            .map(|arr| PrimitiveArray::from_data_default(arr.buffer().clone(), None).boxed())
            .collect::<Vec<_>>();
        Ok(Series::try_from((self.name(), buffers))
            .map_err(PyPolarsErr::from)?
            .into())
    }

    fn get_ptr(&self) -> PyResult<usize> {
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
                let (slice, start, _len) = arr.values().as_slice();
                if start == 0 {
                    Ok(slice.as_ptr() as usize)
                } else {
                    let msg = "Cannot take pointer boolean buffer as it is not perfectly aligned.";
                    raise_err!(msg, ComputeError);
                }
            }
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                get_ptr(ca)
            })),
            _ => {
                let msg = "Cannot take pointer of nested type";
                raise_err!(msg, ComputeError);
            }
        }
    }
}

fn get_ptr<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
}
