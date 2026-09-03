//! The builder of a [`PlPrimitiveArray`].

use arrow::bitmap::OptBitmapBuilder;
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::vec::PushUnchecked;

use super::PlPrimitiveArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlPrimitiveArray`].
pub struct PlPrimitiveArrayBuilder<T: NativeType> {
    values: Vec<T>,
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

    /// Appends the `length` values of `other` starting at `start`, ignoring its validity mask.
    ///
    /// A scalar values buffer is not materialized to be read: the one value it holds is the value
    /// of every element the subslice covers.
    fn extend_values(&mut self, other: &PlPrimitiveArray<T>, start: usize, length: usize) {
        if let Some(values) = other.flat_values() {
            self.values
                .extend_from_slice(&values[start..start + length]);
        } else if let Some(value) = other.scalar_values() {
            self.values.resize(self.values.len() + length, value);
        }
        // An empty array is neither, and the subslice it admits covers no element to append.
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
                Buffer::from(self.values),
                length,
                self.validity.into_opt_validity(),
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
                Buffer::from(values),
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so anything at all does.
        self.values.resize(self.values.len() + length, T::default());
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
        self.values.reserve(length * repeats);

        if let Some(values) = other.flat_values() {
            for value in &values[start..start + length] {
                // SAFETY: room for every repeat of every value was just reserved.
                unsafe {
                    for _ in 0..repeats {
                        self.values.push_unchecked(*value);
                    }
                }
            }
        } else if let Some(value) = other.scalar_values() {
            // Every element repeats the same value, so which of them is repeated is immaterial.
            self.values
                .resize(self.values.len() + length * repeats, value);
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
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        if let Some(values) = other.flat_values() {
            let values = values.as_slice();
            // SAFETY: the indices are in bounds of the array, whose values are flat.
            self.values.extend(
                idxs.iter()
                    .map(|idx| unsafe { *values.get_unchecked(*idx as usize) }),
            );
        } else if let Some(value) = other.scalar_values() {
            // Every index reads the one value the array holds.
            self.values.resize(self.values.len() + idxs.len(), value);
        }
        // An empty array is neither, and admits no index to gather.

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.values.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            let value = if idx < other.len() {
                // SAFETY: the index is in bounds of the array, so it broadcasts into the values.
                unsafe { other.value_unchecked(idx) }
            } else {
                // The value of a null element is undetermined, so anything at all does.
                T::default()
            };
            // SAFETY: room for one value per index was just reserved.
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

    #[test]
    fn scalar_values_are_read_through_the_broadcast() {
        let array = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000)
            .with_validity_broadcast(Some(Bitmap::from_iter([true])));

        let mut builder = PlPrimitiveArrayBuilder::<i32>::with_capacity(8);
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [Some(7), Some(7), Some(7), Some(7), Some(7), Some(7), None],
        );
    }
}
