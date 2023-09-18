use std::sync::Arc;

use crate::{
    array::{Array, NullArray},
    datatypes::DataType,
};

use super::Growable;

/// Concrete [`Growable`] for the [`NullArray`].
pub struct GrowableNull {
    data_type: DataType,
    length: usize,
}

impl Default for GrowableNull {
    fn default() -> Self {
        Self::new(DataType::Null)
    }
}

impl GrowableNull {
    /// Creates a new [`GrowableNull`].
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            length: 0,
        }
    }
}

impl<'a> Growable<'a> for GrowableNull {
    fn extend(&mut self, _: usize, _: usize, len: usize) {
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
