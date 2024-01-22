use std::sync::Arc;

use polars_utils::slice::GetSaferUnchecked;

use super::Growable;
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::{Array, FixedSizeBinaryArray};
use crate::bitmap::MutableBitmap;

/// Concrete [`Growable`] for the [`FixedSizeBinaryArray`].
pub struct GrowableFixedSizeBinary<'a> {
    arrays: Vec<&'a FixedSizeBinaryArray>,
    validity: Option<MutableBitmap>,
    values: Vec<u8>,
    size: usize, // just a cache
}

impl<'a> GrowableFixedSizeBinary<'a> {
    /// Creates a new [`GrowableFixedSizeBinary`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(
        arrays: Vec<&'a FixedSizeBinaryArray>,
        mut use_validity: bool,
        capacity: usize,
    ) -> Self {
        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let size = FixedSizeBinaryArray::get_size(arrays[0].data_type());
        Self {
            arrays,
            values: Vec::with_capacity(0),
            validity: prepare_validity(use_validity, capacity),
            size,
        }
    }

    fn to(&mut self) -> FixedSizeBinaryArray {
        let validity = std::mem::take(&mut self.validity);
        let values = std::mem::take(&mut self.values);

        FixedSizeBinaryArray::new(
            self.arrays[0].data_type().clone(),
            values.into(),
            validity.map(|v| v.into()),
        )
    }
}

impl<'a> Growable<'a> for GrowableFixedSizeBinary<'a> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked_release(index);
        extend_validity(&mut self.validity, array, start, len);

        let values = array.values();

        self.values.extend_from_slice(
            values.get_unchecked_release(start * self.size..start * self.size + len * self.size),
        );
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values
            .extend_from_slice(&vec![0; self.size * additional]);
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len() / self.size
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(self.to())
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a> From<GrowableFixedSizeBinary<'a>> for FixedSizeBinaryArray {
    fn from(val: GrowableFixedSizeBinary<'a>) -> Self {
        FixedSizeBinaryArray::new(
            val.arrays[0].data_type().clone(),
            val.values.into(),
            val.validity.map(|v| v.into()),
        )
    }
}
