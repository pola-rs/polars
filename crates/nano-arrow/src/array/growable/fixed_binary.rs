use std::sync::Arc;

use crate::{
    array::{Array, FixedSizeBinaryArray},
    bitmap::MutableBitmap,
};

use super::{
    utils::{build_extend_null_bits, ExtendNullBits},
    Growable,
};

/// Concrete [`Growable`] for the [`FixedSizeBinaryArray`].
pub struct GrowableFixedSizeBinary<'a> {
    arrays: Vec<&'a FixedSizeBinaryArray>,
    validity: MutableBitmap,
    values: Vec<u8>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
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

        let extend_null_bits = arrays
            .iter()
            .map(|array| build_extend_null_bits(*array, use_validity))
            .collect();

        let size = FixedSizeBinaryArray::get_size(arrays[0].data_type());
        Self {
            arrays,
            values: Vec::with_capacity(0),
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
            size,
        }
    }

    fn to(&mut self) -> FixedSizeBinaryArray {
        let validity = std::mem::take(&mut self.validity);
        let values = std::mem::take(&mut self.values);

        FixedSizeBinaryArray::new(
            self.arrays[0].data_type().clone(),
            values.into(),
            validity.into(),
        )
    }
}

impl<'a> Growable<'a> for GrowableFixedSizeBinary<'a> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);

        let array = self.arrays[index];
        let values = array.values();

        self.values
            .extend_from_slice(&values[start * self.size..start * self.size + len * self.size]);
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values
            .extend_from_slice(&vec![0; self.size * additional]);
        self.validity.extend_constant(additional, false);
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
            val.validity.into(),
        )
    }
}
