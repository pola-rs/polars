use std::sync::Arc;

use super::Growable;
use crate::array::{Array, NullArray};
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`NullArray`].
pub struct GrowableNull {
    dtype: ArrowDataType,
    length: usize,
}

impl Default for GrowableNull {
    fn default() -> Self {
        Self::new(ArrowDataType::Null)
    }
}

impl GrowableNull {
    /// Creates a new [`GrowableNull`].
    pub fn new(dtype: ArrowDataType) -> Self {
        Self { dtype, length: 0 }
    }
}

impl Growable<'_> for GrowableNull {
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
        Arc::new(NullArray::new(self.dtype.clone(), self.length))
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(NullArray::new(self.dtype.clone(), self.length))
    }
}

impl From<GrowableNull> for NullArray {
    fn from(val: GrowableNull) -> Self {
        NullArray::new(val.dtype, val.length)
    }
}
