use std::sync::Arc;

use super::Growable;
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::{Array, BooleanArray};
use crate::bitmap::BitmapBuilder;
use crate::datatypes::ArrowDataType;

/// Concrete [`Growable`] for the [`BooleanArray`].
pub struct GrowableBoolean<'a> {
    arrays: Vec<&'a BooleanArray>,
    dtype: ArrowDataType,
    validity: Option<BitmapBuilder>,
    values: BitmapBuilder,
}

impl<'a> GrowableBoolean<'a> {
    /// Creates a new [`GrowableBoolean`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a BooleanArray>, mut use_validity: bool, capacity: usize) -> Self {
        let dtype = arrays[0].dtype().clone();

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        Self {
            arrays,
            dtype,
            values: BitmapBuilder::with_capacity(capacity),
            validity: prepare_validity(use_validity, capacity),
        }
    }

    fn to(&mut self) -> BooleanArray {
        let validity = self.validity.take();
        let values = std::mem::take(&mut self.values);

        BooleanArray::new(
            self.dtype.clone(),
            values.freeze(),
            validity.map(|v| v.freeze()),
        )
    }
}

impl<'a> Growable<'a> for GrowableBoolean<'a> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);
        extend_validity(&mut self.validity, array, start, len);

        let values = array.values();

        let (slice, offset, _) = values.as_slice();
        self.values.extend_from_slice(slice, start + offset, len);
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values.extend_constant(additional, false);
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(self.to())
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a> From<GrowableBoolean<'a>> for BooleanArray {
    fn from(val: GrowableBoolean<'a>) -> Self {
        BooleanArray::new(
            val.dtype,
            val.values.freeze(),
            val.validity.map(|v| v.freeze()),
        )
    }
}
