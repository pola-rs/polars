//! The builder of a [`PlPrimitiveArray`].

use arrow::bitmap::OptBitmapBuilder;
use arrow::types::{AlignedBytes, NativeType};
use polars_utils::IdxSize;

use super::PlPrimitiveArray;
use super::bytes::{self, Bytes};
use crate::bitmap::PlBitmap;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlPrimitiveArray`].
#[derive(Clone)]
pub struct PlPrimitiveArrayBuilder<T: NativeType> {
    values: Vec<Bytes<T>>,
    validity: OptBitmapBuilder,
}

impl<T: NativeType> PlPrimitiveArrayBuilder<T> {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        builder.reserve(capacity);
        builder
    }

    /// A builder holding `values` and `validity` as the elements appended so far.
    pub(crate) fn from_parts(values: Vec<Bytes<T>>, validity: OptBitmapBuilder) -> Self {
        Self { values, validity }
    }

    /// Appends `value` as an element of its own.
    #[inline]
    pub fn push_value(&mut self, value: T) {
        self.values.push(bytes::to_bytes(value));
        self.validity.extend_constant(1, true);
    }

    /// Appends every value `values` yields, in order, none of them null.
    #[inline]
    pub fn push_values<I: IntoIterator<Item = T>>(&mut self, values: I) {
        let before = self.values.len();
        self.values
            .extend(values.into_iter().map(bytes::to_bytes::<T>));
        self.validity
            .extend_constant(self.values.len() - before, true);
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        // The value of a null element is undetermined, so anything at all does.
        self.values.push(Bytes::<T>::zeros());
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    #[inline]
    pub fn push(&mut self, value: Option<T>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Appends the `length` values of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(&mut self, other: &PlPrimitiveArray<T>, start: usize, length: usize) {
        bytes::extend_subslice(&mut self.values, other.values_bytes(), start, length);
    }
}

impl<T: NativeType> Default for PlPrimitiveArrayBuilder<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativeType> StaticArrayBuilder for PlPrimitiveArrayBuilder<T> {
    type Array = PlPrimitiveArray<T>;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    fn freeze(self) -> PlPrimitiveArray<T> {
        let length = self.values.len();
        // SAFETY: the values hold one slot per element, and so does the mask that was built
        // alongside them.
        unsafe {
            PlPrimitiveArray::new_unchecked(
                bytes::buffer_from_byte_vec::<T>(self.values),
                length,
                self.validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn freeze_reset(&mut self) -> PlPrimitiveArray<T> {
        let values = std::mem::take(&mut self.values);
        let validity = std::mem::take(&mut self.validity);
        let length = values.len();
        // SAFETY: as in `freeze`.
        unsafe {
            PlPrimitiveArray::new_unchecked(
                bytes::buffer_from_byte_vec::<T>(values),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so anything at all does.
        bytes::extend_undetermined(&mut self.values, length);
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlPrimitiveArray<T>,
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
        other: &PlPrimitiveArray<T>,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        bytes::extend_subslice_each_repeated(
            &mut self.values,
            other.values_bytes(),
            start,
            length,
            repeats,
        );

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
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of the array, and therefore of its values.
        unsafe { bytes::extend_gathered(&mut self.values, other.values_bytes(), idxs) };

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        bytes::extend_opt_gathered(&mut self.values, other.values_bytes(), other.len(), idxs);

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn appending_subslices_and_repeats() {
        let array: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();

        let mut builder = PlPrimitiveArrayBuilder::<i32>::new();
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Never);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 0, 2, 2, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [
                None,
                Some(3),
                Some(1),
                None,
                Some(1),
                None,
                Some(1),
                Some(1),
                None,
                None,
            ],
        );
    }

    #[test]
    fn gathering() {
        let array: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();

        let mut builder = PlPrimitiveArrayBuilder::<i32>::new();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Never) };
        builder.opt_gather_extend(&array, &[1, 7], ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [Some(3), Some(1), None, None, None],
        );
    }

    /// Appending a run has to leave the mask exactly as long as the values, whether or not a
    /// mask has come into being yet — the two are extended by separate calls.
    #[test]
    fn pushing_a_run_of_values_keeps_the_mask_aligned() {
        let mut builder = PlPrimitiveArrayBuilder::<i32>::new();
        builder.push_values(0..4);
        builder.push_null();
        builder.push_values([9, 8]);
        builder.push_value(7);

        let built = builder.freeze();
        assert_eq!(built.len(), 8);
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                None,
                Some(9),
                Some(8),
                Some(7),
            ],
        );
    }
}
