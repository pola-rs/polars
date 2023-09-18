use std::sync::Arc;

use crate::{
    array::{Array, BooleanArray},
    bitmap::MutableBitmap,
    datatypes::DataType,
};

use super::{
    utils::{build_extend_null_bits, ExtendNullBits},
    Growable,
};

/// Concrete [`Growable`] for the [`BooleanArray`].
pub struct GrowableBoolean<'a> {
    arrays: Vec<&'a BooleanArray>,
    data_type: DataType,
    validity: MutableBitmap,
    values: MutableBitmap,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
}

impl<'a> GrowableBoolean<'a> {
    /// Creates a new [`GrowableBoolean`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a BooleanArray>, mut use_validity: bool, capacity: usize) -> Self {
        let data_type = arrays[0].data_type().clone();

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let extend_null_bits = arrays
            .iter()
            .map(|array| build_extend_null_bits(*array, use_validity))
            .collect();

        Self {
            arrays,
            data_type,
            values: MutableBitmap::with_capacity(capacity),
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
        }
    }

    fn to(&mut self) -> BooleanArray {
        let validity = std::mem::take(&mut self.validity);
        let values = std::mem::take(&mut self.values);

        BooleanArray::new(self.data_type.clone(), values.into(), validity.into())
    }
}

impl<'a> Growable<'a> for GrowableBoolean<'a> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);

        let array = self.arrays[index];
        let values = array.values();

        let (slice, offset, _) = values.as_slice();
        // safety: invariant offset + length <= slice.len()
        unsafe {
            self.values
                .extend_from_slice_unchecked(slice, start + offset, len);
        }
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values.extend_constant(additional, false);
        self.validity.extend_constant(additional, false);
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
        BooleanArray::new(val.data_type, val.values.into(), val.validity.into())
    }
}
