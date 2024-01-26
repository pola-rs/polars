use std::sync::Arc;

use polars_utils::slice::GetSaferUnchecked;

use super::utils::extend_offset_values;
use super::Growable;
use crate::array::growable::utils::{extend_validity, prepare_validity};
use crate::array::{Array, Utf8Array};
use crate::bitmap::MutableBitmap;
use crate::offset::{Offset, Offsets};

/// Concrete [`Growable`] for the [`Utf8Array`].
pub struct GrowableUtf8<'a, O: Offset> {
    arrays: Vec<&'a Utf8Array<O>>,
    validity: Option<MutableBitmap>,
    values: Vec<u8>,
    offsets: Offsets<O>,
}

impl<'a, O: Offset> GrowableUtf8<'a, O> {
    /// Creates a new [`GrowableUtf8`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a Utf8Array<O>>, mut use_validity: bool, capacity: usize) -> Self {
        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        Self {
            arrays: arrays.to_vec(),
            values: Vec::with_capacity(0),
            offsets: Offsets::with_capacity(capacity),
            validity: prepare_validity(use_validity, capacity),
        }
    }

    fn to(&mut self) -> Utf8Array<O> {
        let validity = std::mem::take(&mut self.validity);
        let offsets = std::mem::take(&mut self.offsets);
        let values = std::mem::take(&mut self.values);

        #[cfg(debug_assertions)]
        {
            crate::array::specification::try_check_utf8(offsets.as_slice(), &values).unwrap();
        }

        unsafe {
            Utf8Array::<O>::new_unchecked(
                self.arrays[0].data_type().clone(),
                offsets.into(),
                values.into(),
                validity.map(|v| v.into()),
            )
        }
    }
}

impl<'a, O: Offset> Growable<'a> for GrowableUtf8<'a, O> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked_release(index);
        extend_validity(&mut self.validity, array, start, len);

        let offsets = array.offsets();
        let values = array.values();

        self.offsets
            .try_extend_from_slice(offsets, start, len)
            .unwrap();

        // values
        extend_offset_values::<O>(&mut self.values, offsets.as_slice(), values, start, len);
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
        Arc::new(self.to())
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a, O: Offset> From<GrowableUtf8<'a, O>> for Utf8Array<O> {
    fn from(mut val: GrowableUtf8<'a, O>) -> Self {
        val.to()
    }
}
