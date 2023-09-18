use std::sync::Arc;

use crate::{
    array::{Array, BinaryArray},
    bitmap::MutableBitmap,
    datatypes::DataType,
    offset::{Offset, Offsets},
};

use super::{
    utils::{build_extend_null_bits, extend_offset_values, ExtendNullBits},
    Growable,
};

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinary<'a, O: Offset> {
    arrays: Vec<&'a BinaryArray<O>>,
    data_type: DataType,
    validity: MutableBitmap,
    values: Vec<u8>,
    offsets: Offsets<O>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
}

impl<'a, O: Offset> GrowableBinary<'a, O> {
    /// Creates a new [`GrowableBinary`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a BinaryArray<O>>, mut use_validity: bool, capacity: usize) -> Self {
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
            values: Vec::with_capacity(0),
            offsets: Offsets::with_capacity(capacity),
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
        }
    }

    fn to(&mut self) -> BinaryArray<O> {
        let data_type = self.data_type.clone();
        let validity = std::mem::take(&mut self.validity);
        let offsets = std::mem::take(&mut self.offsets);
        let values = std::mem::take(&mut self.values);

        BinaryArray::<O>::new(data_type, offsets.into(), values.into(), validity.into())
    }
}

impl<'a, O: Offset> Growable<'a> for GrowableBinary<'a, O> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);

        let array = self.arrays[index];
        let offsets = array.offsets();
        let values = array.values();

        self.offsets
            .try_extend_from_slice(offsets, start, len)
            .unwrap();

        // values
        extend_offset_values::<O>(&mut self.values, offsets.buffer(), values, start, len);
    }

    fn extend_validity(&mut self, additional: usize) {
        self.offsets.extend_constant(additional);
        self.validity.extend_constant(additional, false);
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        self.to().arced()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.to().boxed()
    }
}

impl<'a, O: Offset> From<GrowableBinary<'a, O>> for BinaryArray<O> {
    fn from(val: GrowableBinary<'a, O>) -> Self {
        BinaryArray::<O>::new(
            val.data_type,
            val.offsets.into(),
            val.values.into(),
            val.validity.into(),
        )
    }
}
