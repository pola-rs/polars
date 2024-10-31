use arrow::array::{Array, MapArray, MutableArray};
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableMapArray {
    dtype: ArrowDataType,
    pub inner: Box<dyn MutableArray>,
}

impl DynMutableMapArray {
    pub fn try_with_capacity(dtype: ArrowDataType, capacity: usize) -> PolarsResult<Self> {
        let inner = match dtype.to_logical_type() {
            ArrowDataType::Map(inner, _) => inner,
            _ => unreachable!(),
        };
        let inner = make_mutable(inner.dtype(), capacity)?;

        Ok(Self { dtype, inner })
    }
}

impl MutableArray for DynMutableMapArray {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&arrow::bitmap::MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(MapArray::new(
            self.dtype.clone(),
            vec![0, self.inner.len() as i32].try_into().unwrap(),
            self.inner.as_box(),
            None,
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn push_null(&mut self) {
        todo!()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!()
    }
}
