use std::any::Any;

use crate::array::{Array, MutableArray, NullArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;

#[derive(Debug, Default, Clone)]
pub struct MutableNullArray {
    len: usize,
}

impl MutableArray for MutableNullArray {
    fn data_type(&self) -> &ArrowDataType {
        &ArrowDataType::Null
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(NullArray::new_null(ArrowDataType::Null, self.len))
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

impl MutableNullArray {
    pub fn new(len: usize) -> Self {
        MutableNullArray { len }
    }

    pub fn extend_nulls(&mut self, null_count: usize) {
        self.len += null_count;
    }
}
