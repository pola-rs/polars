use std::sync::Arc;

use crate::{
    array::{Array, ListArray},
    bitmap::MutableBitmap,
    offset::{Offset, Offsets},
};

use super::{
    make_growable,
    utils::{build_extend_null_bits, ExtendNullBits},
    Growable,
};

fn extend_offset_values<O: Offset>(
    growable: &mut GrowableList<'_, O>,
    index: usize,
    start: usize,
    len: usize,
) {
    let array = growable.arrays[index];
    let offsets = array.offsets();

    growable
        .offsets
        .try_extend_from_slice(offsets, start, len)
        .unwrap();

    let end = offsets.buffer()[start + len].to_usize();
    let start = offsets.buffer()[start].to_usize();
    let len = end - start;
    growable.values.extend(index, start, len);
}

/// Concrete [`Growable`] for the [`ListArray`].
pub struct GrowableList<'a, O: Offset> {
    arrays: Vec<&'a ListArray<O>>,
    validity: MutableBitmap,
    values: Box<dyn Growable<'a> + 'a>,
    offsets: Offsets<O>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
}

impl<'a, O: Offset> GrowableList<'a, O> {
    /// Creates a new [`GrowableList`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(arrays: Vec<&'a ListArray<O>>, mut use_validity: bool, capacity: usize) -> Self {
        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let extend_null_bits = arrays
            .iter()
            .map(|array| build_extend_null_bits(*array, use_validity))
            .collect();

        let inner = arrays
            .iter()
            .map(|array| array.values().as_ref())
            .collect::<Vec<_>>();
        let values = make_growable(&inner, use_validity, 0);

        Self {
            arrays,
            offsets: Offsets::with_capacity(capacity),
            values,
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
        }
    }

    fn to(&mut self) -> ListArray<O> {
        let validity = std::mem::take(&mut self.validity);
        let offsets = std::mem::take(&mut self.offsets);
        let values = self.values.as_box();

        ListArray::<O>::new(
            self.arrays[0].data_type().clone(),
            offsets.into(),
            values,
            validity.into(),
        )
    }
}

impl<'a, O: Offset> Growable<'a> for GrowableList<'a, O> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);
        extend_offset_values::<O>(self, index, start, len);
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
        Arc::new(self.to())
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a, O: Offset> From<GrowableList<'a, O>> for ListArray<O> {
    fn from(mut val: GrowableList<'a, O>) -> Self {
        val.to()
    }
}
