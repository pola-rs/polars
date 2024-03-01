use std::sync::Arc;

use super::Growable;
use crate::array::{Array, NullArray};
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`NullArray`].
pub struct GrowableNull {
    data_type: ArrowDataType,
    length: usize,
}

impl Default for GrowableNull {
    fn default() -> Self {
        Self::new(ArrowDataType::Null)
    }
}

impl GrowableNull {
    /// Creates a new [`GrowableNull`].
    pub fn new(data_type: ArrowDataType) -> Self {
        Self {
            data_type,
            length: 0,
        }
    }
}

impl<'a> Growable<'a> for GrowableNull {
    unsafe fn extend(&mut self, _: usize, _: usize, len: usize) {
        self.length += len;
    }

    fn extend_validity(&mut self, additional: usize) {
        self.length += additional;
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(NullArray::new(self.data_type.clone(), self.length))
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(NullArray::new(self.data_type.clone(), self.length))
    }
}

impl From<GrowableNull> for NullArray {
    fn from(val: GrowableNull) -> Self {
        NullArray::new(val.data_type, val.length)
    }
}
