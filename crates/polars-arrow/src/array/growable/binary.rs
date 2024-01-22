use std::sync::Arc;

use polars_utils::slice::GetSaferUnchecked;

use super::utils::extend_offset_values;
use super::Growable;
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::{Array, BinaryArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::offset::{Offset, Offsets};

/// Concrete [`Growable`] for the [`BinaryArray`].
pub struct GrowableBinary<'a, O: Offset> {
    arrays: Vec<&'a BinaryArray<O>>,
    data_type: ArrowDataType,
    validity: Option<MutableBitmap>,
    values: Vec<u8>,
    offsets: Offsets<O>,
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

        Self {
            arrays,
            data_type,
            values: Vec::with_capacity(0),
            offsets: Offsets::with_capacity(capacity),
            validity: prepare_validity(use_validity, capacity),
        }
    }

    fn to(&mut self) -> BinaryArray<O> {
        let data_type = self.data_type.clone();
        let validity = std::mem::take(&mut self.validity);
        let offsets = std::mem::take(&mut self.offsets);
        let values = std::mem::take(&mut self.values);

        BinaryArray::<O>::new(
            data_type,
            offsets.into(),
            values.into(),
            validity.map(|v| v.into()),
        )
    }
}

impl<'a, O: Offset> Growable<'a> for GrowableBinary<'a, O> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked_release(index);
        extend_validity(&mut self.validity, array, start, len);

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
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
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
            val.validity.map(|v| v.into()),
        )
    }
}
