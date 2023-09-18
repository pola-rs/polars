use std::sync::Arc;

use crate::{
    array::{Array, FixedSizeListArray},
    bitmap::MutableBitmap,
    datatypes::DataType,
};

use super::{
    make_growable,
    utils::{build_extend_null_bits, ExtendNullBits},
    Growable,
};

/// Concrete [`Growable`] for the [`FixedSizeListArray`].
pub struct GrowableFixedSizeList<'a> {
    arrays: Vec<&'a FixedSizeListArray>,
    validity: MutableBitmap,
    values: Box<dyn Growable<'a> + 'a>,
    extend_null_bits: Vec<ExtendNullBits<'a>>,
    size: usize,
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

        let size =
            if let DataType::FixedSizeList(_, size) = &arrays[0].data_type().to_logical_type() {
                *size
            } else {
                unreachable!("`GrowableFixedSizeList` expects `DataType::FixedSizeList`")
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
            values,
            validity: MutableBitmap::with_capacity(capacity),
            extend_null_bits,
            size,
        }
    }

    fn to(&mut self) -> FixedSizeListArray {
        let validity = std::mem::take(&mut self.validity);
        let values = self.values.as_box();

        FixedSizeListArray::new(self.arrays[0].data_type().clone(), values, validity.into())
    }
}

impl<'a> Growable<'a> for GrowableFixedSizeList<'a> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        (self.extend_null_bits[index])(&mut self.validity, start, len);
        self.values
            .extend(index, start * self.size, len * self.size);
    }

    fn extend_validity(&mut self, additional: usize) {
        self.values.extend_validity(additional * self.size);
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

impl<'a> From<GrowableFixedSizeList<'a>> for FixedSizeListArray {
    fn from(val: GrowableFixedSizeList<'a>) -> Self {
        let mut values = val.values;
        let values = values.as_box();

        Self::new(
            val.arrays[0].data_type().clone(),
            values,
            val.validity.into(),
        )
    }
}
