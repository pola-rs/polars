//! The builder of a [`PlBooleanArray`].

use arrow::bitmap::{BitmapBuilder, OptBitmapBuilder};
use polars_utils::IdxSize;

use super::PlBooleanArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBooleanArray`].
pub struct PlBooleanArrayBuilder {
    values: BitmapBuilder,
    validity: OptBitmapBuilder,
}

impl PlBooleanArrayBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            values: BitmapBuilder::new(),
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        builder.reserve(capacity);
        builder
    }

    /// Appends `value` as an element of its own.
    #[inline]
    pub fn push_value(&mut self, value: bool) {
        self.values.push(value);
        self.validity.extend_constant(1, true);
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        // The value of a null element is undetermined, so anything at all does.
        self.values.push(false);
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    #[inline]
    pub fn push(&mut self, value: Option<bool>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Appends the `length` values of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(&mut self, other: &PlBooleanArray, start: usize, length: usize) {
        if let Some(values) = other.flat_values() {
            self.values
                .subslice_extend_from_bitmap(values, start, length);
        } else if let Some(value) = other.scalar_values() {
            self.values.extend_constant(length, value);
        }
        // An empty array is neither, and the subslice it admits covers no element to append.
    }
}

impl Default for PlBooleanArrayBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StaticArrayBuilder for PlBooleanArrayBuilder {
    type Array = PlBooleanArray;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    fn freeze(self) -> PlBooleanArray {
        let length = self.values.len();
        let validity = self.validity.into_opt_validity();
        // SAFETY: the values hold one bit per element, and so does the mask that was built
        // alongside them.
        unsafe { PlBooleanArray::new_unchecked(self.values.freeze(), length, validity) }
    }

    fn freeze_reset(&mut self) -> PlBooleanArray {
        let values = std::mem::take(&mut self.values);
        let validity = std::mem::take(&mut self.validity);
        let length = values.len();
        // SAFETY: as in `freeze`.
        unsafe {
            PlBooleanArray::new_unchecked(values.freeze(), length, validity.into_opt_validity())
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so anything at all does.
        self.values.extend_constant(length, false);
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlBooleanArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        self.extend_values(other, start, length);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlBooleanArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.values.reserve(length * repeats);

        if let Some(values) = other.flat_values() {
            self.values
                .subslice_extend_each_repeated_from_bitmap(values, start, length, repeats);
        } else if let Some(value) = other.scalar_values() {
            // Every element repeats the same value, so which of them is repeated is immaterial.
            self.values.extend_constant(length * repeats, value);
        }
        // An empty array is neither, and the subslice it admits covers no element to append.

        subslice_extend_each_repeated_validity(
            &mut self.validity,
            other.validity(),
            start,
            length,
            repeats,
        );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PlBooleanArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.values.reserve(idxs.len());

        match other.values().scalar_value() {
            // Every index reads the one bit the array holds.
            Some(value) => self.values.extend_constant(idxs.len(), value),
            None => {
                for idx in idxs {
                    // SAFETY: the indices are in bounds of the array, and room for one bit per
                    // index was just reserved.
                    unsafe {
                        self.values
                            .push_unchecked(other.value_unchecked(*idx as usize));
                    }
                }
            },
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlBooleanArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.values.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            // The value of a null element is undetermined, so an out-of-bounds index writes
            // anything at all.
            let value = idx < other.len() && unsafe { other.value_unchecked(idx) };
            // SAFETY: room for one bit per index was just reserved.
            unsafe { self.values.push_unchecked(value) };
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    #[test]
    fn appending_subslices_and_repeats() {
        let array: PlBooleanArray = [Some(true), None, Some(false)].into_iter().collect();

        let mut builder = PlBooleanArrayBuilder::with_capacity(8);
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Never);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [
                None,
                Some(false),
                Some(true),
                None,
                Some(true),
                None,
                Some(false),
                Some(false),
            ],
        );
    }

    #[test]
    fn gathering() {
        let array: PlBooleanArray = [Some(true), None, Some(false)].into_iter().collect();

        let mut builder = PlBooleanArrayBuilder::new();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Never) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [Some(false), Some(true), None, Some(true), None],
        );
    }

    #[test]
    fn pushing_elements_one_at_a_time() {
        let mut builder = PlBooleanArrayBuilder::with_capacity(4);
        builder.push_value(true);
        builder.push_null();
        builder.push(Some(false));
        builder.push(None);

        assert_eq!(builder.len(), 4);
        assert_eq!(
            builder.freeze().iter().collect::<Vec<_>>(),
            [Some(true), None, Some(false), None],
        );

        // The mask only comes into being once a null is pushed.
        let mut valid = PlBooleanArrayBuilder::new();
        valid.push_value(true);
        assert!(valid.freeze().validity().is_none());
    }

    #[test]
    fn scalar_values_are_read_through_the_broadcast() {
        let array = PlBooleanArray::new_scalar(true, 1_000_000_000)
            .with_validity_broadcast(Some(Bitmap::from_iter([true])));

        let mut builder = PlBooleanArrayBuilder::new();
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 7);
        assert_eq!(built.null_count(), 1);
        assert_eq!(built.iter().take(6).flatten().count(), 6);
    }
}
