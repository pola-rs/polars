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
}
