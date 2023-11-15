use arrow::array::{Array, MapArray, MutableArray};
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableMapArray {
    data_type: ArrowDataType,
    pub inner: Box<dyn MutableArray>,
}

impl DynMutableMapArray {
    pub fn try_with_capacity(data_type: ArrowDataType, capacity: usize) -> PolarsResult<Self> {
        let inner = match data_type.to_logical_type() {
            ArrowDataType::Map(inner, _) => inner,
            _ => unreachable!(),
        };
        let inner = make_mutable(inner.data_type(), capacity)?;

        Ok(Self { data_type, inner })
    }
}

impl MutableArray for DynMutableMapArray {
    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&arrow::bitmap::MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(MapArray::new(
            self.data_type.clone(),
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
