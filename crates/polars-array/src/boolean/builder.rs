//! The builder of a [`PlBooleanArray`].

use arrow::bitmap::{BitmapBuilder, OptBitmapBuilder};
use polars_utils::IdxSize;

use super::PlBooleanArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBooleanArray`].
///
/// The values are staged in a [`BitmapBuilder`], which is what the frozen array is taken over, so
/// the array this builds is [flat](crate::Flat) — a bit per element, however many of the appended
/// elements shared one. The value of a null element is written out as `false`.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder};
/// use polars_array::{PlBooleanArray, PlBooleanArrayBuilder};
///
/// let mut builder = PlBooleanArrayBuilder::new();
/// builder.extend_nulls(1);
/// builder.extend(&PlBooleanArray::new_scalar(true, 2), ShareStrategy::Never);
///
/// let array = builder.freeze();
/// assert_eq!(array.iter().collect::<Vec<_>>(), [None, Some(true), Some(true)]);
/// ```
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

    /// Appends the `length` values of `other` starting at `start`, ignoring its validity mask.
    ///
    /// A scalar values bitmap is not materialized to be read: the one bit it holds is the value of
    /// every element the subslice covers.
    fn extend_values(&mut self, other: &PlBooleanArray, start: usize, length: usize) {
        let values = other.values();
        match values.scalar_value() {
            Some(value) => self.values.extend_constant(length, value),
            None => self
                .values
                .subslice_extend_from_bitmap(values.bitmap(), start, length),
        }
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

        let values = other.values();
        match values.scalar_value() {
            // Every element repeats the same value, so which of them is repeated is immaterial.
            Some(value) => self.values.extend_constant(length * repeats, value),
            None => self.values.subslice_extend_each_repeated_from_bitmap(
                values.bitmap(),
                start,
                length,
                repeats,
            ),
        }

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
    fn scalar_values_are_read_through_the_broadcast() {
        let array = PlBooleanArray::new_scalar(true, 1_000_000_000)
            .with_validity(Some(Bitmap::from_iter([true])));

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

    #[test]
    fn freeze_reset_leaves_an_empty_builder() {
        let mut builder = PlBooleanArrayBuilder::new();
        builder.extend(
            &PlBooleanArray::from_vec(vec![true, false]),
            ShareStrategy::Never,
        );

        assert_eq!(builder.freeze_reset().len(), 2);
        assert!(builder.is_empty());
        assert!(builder.freeze().is_empty());
    }
}
