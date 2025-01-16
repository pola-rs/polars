use std::sync::Arc;

use super::{make_growable, Growable};
use crate::array::growable::utils::{extend_validity, extend_validity_copies, prepare_validity};
use crate::array::{Array, FixedSizeListArray};
use crate::bitmap::BitmapBuilder;

/// Concrete [`Growable`] for the [`FixedSizeListArray`].
pub struct GrowableFixedSizeList<'a> {
    arrays: Vec<&'a FixedSizeListArray>,
    validity: Option<BitmapBuilder>,
    values: Box<dyn Growable<'a> + 'a>,
    size: usize,
    length: usize,
}

impl<'a> GrowableFixedSizeList<'a> {
    /// Creates a new [`GrowableFixedSizeList`] bound to `arrays` with a pre-allocated `capacity`.
    /// # Panics
    /// If `arrays` is empty.
    pub fn new(
        arrays: Vec<&'a FixedSizeListArray>,
        mut use_validity: bool,
        capacity: usize,
    ) -> Self {
        assert!(!arrays.is_empty());

        // if any of the arrays has nulls, insertions from any array requires setting bits
        // as there is at least one array with nulls.
        if !use_validity & arrays.iter().any(|array| array.null_count() > 0) {
            use_validity = true;
        };

        let size = arrays[0].size();

        let inner = arrays
            .iter()
            .map(|array| {
                debug_assert_eq!(array.size(), size);
                array.values().as_ref()
            })
            .collect::<Vec<_>>();
        let values = make_growable(&inner, use_validity, 0);

        assert_eq!(values.len(), 0);

        Self {
            arrays,
            values,
            validity: prepare_validity(use_validity, capacity),
            size,
            length: 0,
        }
    }

    pub fn to(&mut self) -> FixedSizeListArray {
        let validity = std::mem::take(&mut self.validity);
        let values = self.values.as_box();

        FixedSizeListArray::new(
            self.arrays[0].dtype().clone(),
            self.length,
            values,
            validity.map(|v| v.freeze()),
        )
    }
}

impl<'a> Growable<'a> for GrowableFixedSizeList<'a> {
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize) {
        let array = *self.arrays.get_unchecked(index);
        extend_validity(&mut self.validity, array, start, len);

        self.length += len;
        let start_length = self.values.len();
        self.values
            .extend(index, start * self.size, len * self.size);
        debug_assert!(self.size == 0 || (self.values.len() - start_length) / self.size == len);
    }

    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        let array = *self.arrays.get_unchecked(index);
        extend_validity_copies(&mut self.validity, array, start, len, copies);

        self.length += len * copies;
        let start_length = self.values.len();
        self.values
            .extend_copies(index, start * self.size, len * self.size, copies);
        debug_assert!(
            self.size == 0 || (self.values.len() - start_length) / self.size == len * copies
        );
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values.extend_validity(additional * self.size);
        if let Some(validity) = &mut self.validity {
            validity.extend_constant(additional, false);
        }
        self.length += additional;
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        Arc::new(self.to())
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(self.to())
    }
}

impl<'a> From<GrowableFixedSizeList<'a>> for FixedSizeListArray {
    fn from(val: GrowableFixedSizeList<'a>) -> Self {
        let mut values = val.values;
        let values = values.as_box();

        Self::new(
            val.arrays[0].dtype().clone(),
            val.length,
            values,
            val.validity.map(|v| v.freeze()),
        )
    }
}
