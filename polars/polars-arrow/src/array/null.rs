use std::any::Any;

use arrow::array::{Array, MutableArray, NullArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::DataType;

#[derive(Debug, Default)]
pub struct MutableNullArray {
    len: usize,
}

impl MutableArray for MutableNullArray {
    fn data_type(&self) -> &DataType {
        &DataType::Null
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(NullArray::new_null(DataType::Null, self.len))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        self.len += 1;
    }

    fn reserve(&mut self, _additional: usize) {
        // no-op
    }

    fn shrink_to_fit(&mut self) {
        // no-op
    }
}
