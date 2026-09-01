//! The builder of a [`PlListArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlListArray;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlListArray`].
///
/// A list array is offsets over a values array, so this builder is a growing offsets buffer over
/// the builder of those values: appending an element appends the values its list covers to the
/// child builder, and one offset here. The offsets hold one slot per element plus the end of the
/// last, so the array this builds is [flat](crate::Flat) — one range per element, however many of
/// the appended elements shared one.
///
/// The child builder is what the values of the built array come out of, so it is what decides
/// their representation and how they are appended; [`ShareStrategy`] is passed straight through to
/// it. A null element is written out as an empty list, which reaches no values at all.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
/// use polars_array::{PlArray, PlListArray, PlListArrayBuilder, PlPrimitiveArray};
/// use polars_buffer::Buffer;
///
/// // Two lists over three values: `[1, 2]` and `[3]`.
/// let array = PlListArray::from_offsets(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
///     Buffer::from(vec![0u64, 2, 3]),
/// );
///
/// let mut builder = PlListArrayBuilder::new(builder_like(array.values()));
/// builder.extend_nulls(1);
/// builder.extend(&array, ShareStrategy::Always);
///
/// let built = builder.freeze();
/// assert_eq!(built.len(), 3);
/// assert_eq!(built.value_range(0), 0..0);
/// assert_eq!(built.value_range(2), 2..3);
/// assert_eq!(built.values().len(), 3);
/// ```
pub struct PlListArrayBuilder<B: PlArrayBuilder = Box<dyn PlArrayBuilder>> {
    /// The start of every element appended so far, plus the end of the last: one slot more than
    /// the elements, which is what the offsets of a flat list array hold.
    offsets: Vec<u64>,
    values: B,
    validity: OptBitmapBuilder,
}

impl<B: PlArrayBuilder> PlListArrayBuilder<B> {
    /// Creates an empty builder over the builder of the values the lists are taken over.
    pub fn new(values: B) -> Self {
        Self {
            // The end of the last element of an empty array, which is where the first one starts.
            offsets: vec![0],
            values,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(values: B, capacity: usize) -> Self {
        let mut builder = Self::new(values);
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The builder of the values the lists are taken over.
    #[inline]
    pub fn values(&self) -> &B {
        &self.values
    }

    /// The end of the last element appended, which is where the next one starts.
    #[inline]
    fn last_offset(&self) -> u64 {
        // The offsets are never empty: they start out holding the end of no element at all.
        self.offsets[self.offsets.len() - 1]
    }

    /// Appends one element covering the `length` values that were just appended to the child.
    #[inline]
    fn push_offset(&mut self, length: usize) {
        self.offsets.push(self.last_offset() + length as u64);
    }

    /// Appends the `length` elements of `other` starting at `start`, ignoring its validity mask.
    ///
    /// The elements of an array whose offsets are flat are laid end to end, so the values of all
    /// of them are appended in one go and the offsets are the ones `other` already holds, rebased
    /// onto the values appended so far.
    fn extend_values(
        &mut self,
        other: &PlListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        if other.offsets_are_scalar() {
            if length == 0 {
                return;
            }

            // Every element covers the same range, which is appended once per element; the array
            // is not empty, so the first element is the one to read it off.
            let range = other.value_range(0);
            self.values.subslice_extend_repeated(
                other.values(),
                range.start,
                range.len(),
                length,
                share,
            );

            let mut offset = self.last_offset();
            self.offsets.reserve(length);
            for _ in 0..length {
                offset += range.len() as u64;
                self.offsets.push(offset);
            }
            return;
        }

        let offsets = other.offsets();
        let (first, last) = (offsets[start], offsets[start + length]);
        self.values.subslice_extend(
            other.values(),
            first as usize,
            (last - first) as usize,
            share,
        );

        let base = self.last_offset();
        self.offsets
            .extend(offsets[start + 1..=start + length].iter().map(|offset| {
                // The offsets of `other` start at `first`, and the ones here at the end of the
                // last element appended.
                base + (offset - first)
            }));
    }
}

impl<B: PlArrayBuilder> StaticArrayBuilder for PlListArrayBuilder<B> {
    type Array = PlListArray;

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
        // The child is not reserved for: how many values the elements reach is not implied by how
        // many elements there are.
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn freeze(self) -> PlListArray {
        let length = self.offsets.len() - 1;
        let validity = self.validity.into_opt_validity();
        // SAFETY: the offsets hold the start of every element plus the end of the last, they are
        // pushed in non-decreasing order, and they reach exactly the values appended to the child.
        unsafe {
            PlListArray::new_unchecked(
                self.values.freeze(),
                Buffer::from(self.offsets),
                length,
                validity,
            )
        }
    }

    fn freeze_reset(&mut self) -> PlListArray {
        let offsets = std::mem::replace(&mut self.offsets, vec![0]);
        let validity = std::mem::take(&mut self.validity);
        let length = offsets.len() - 1;
        // SAFETY: as in `freeze`.
        unsafe {
            PlListArray::new_unchecked(
                self.values.freeze_reset(),
                Buffer::from(offsets),
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so the empty list every one of them covers
        // reaches no values to append to the child.
        let offset = self.last_offset();
        self.offsets.extend(std::iter::repeat_n(offset, length));
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length);

        self.extend_values(other, start, length, share);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlListArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length * repeats);

        if other.offsets_are_scalar() {
            // Every element covers the same range, so which of them is repeated is immaterial.
            self.extend_values(other, start, length * repeats, share);
        } else {
            for i in start..start + length {
                // SAFETY: `i` is in bounds of the array, whose offsets are flat.
                let range = unsafe { other.value_range_unchecked(i) };
                self.values.subslice_extend_repeated(
                    other.values(),
                    range.start,
                    range.len(),
                    repeats,
                    share,
                );
                for _ in 0..repeats {
                    self.push_offset(range.len());
                }
            }
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
        other: &PlListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.offsets.reserve(idxs.len());

        if other.offsets_are_scalar() {
            // Every index reads the one range the array holds.
            self.extend_values(other, 0, idxs.len(), share);
        } else {
            // A run of consecutive indices is a subslice, which the child appends in one go.
            let mut run_start = 0;
            while run_start < idxs.len() {
                let first = idxs[run_start] as usize;
                let mut run_length = 1;
                while run_start + run_length < idxs.len()
                    && idxs[run_start + run_length] as usize == first + run_length
                {
                    run_length += 1;
                }

                self.extend_values(other, first, run_length, share);
                run_start += run_length;
            }
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(&mut self, other: &PlListArray, idxs: &[IdxSize], share: ShareStrategy) {
        self.offsets.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                let range = unsafe { other.value_range_unchecked(idx) };
                self.values
                    .subslice_extend(other.values(), range.start, range.len(), share);
                self.push_offset(range.len());
            } else {
                // An out-of-bounds index stands for a null, which covers the empty list.
                self.push_offset(0);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_buffer::Buffer;

    use super::PlListArrayBuilder;
    use crate::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
    use crate::{PlListArray, PlPrimitiveArray};

    /// Three lists over five values: `[1, 2]`, null and `[3, 4, 5]`.
    fn array() -> PlListArray {
        PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
            Buffer::from(vec![0u64, 2, 2, 5]),
        )
        .with_validity(Some(Bitmap::from_iter([true, false, true])))
    }

    /// The elements of a list array, as the values of the lists they cover.
    fn elements(array: &PlListArray) -> Vec<Option<Vec<i32>>> {
        array
            .iter()
            .map(|element| {
                element.map(|element| {
                    element
                        .as_any()
                        .downcast_ref::<PlPrimitiveArray<i32>>()
                        .unwrap()
                        .values_iter()
                        .collect()
                })
            })
            .collect()
    }

    fn builder() -> PlListArrayBuilder {
        PlListArrayBuilder::with_capacity(builder_like(array().values()), 8)
    }

    #[test]
    fn appending_subslices_and_repeats() {
        let array = array();

        let mut builder = builder();
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Always);
        builder.extend_nulls(1);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                None,
                Some(vec![3, 4, 5]),
                None,
                Some(vec![1, 2]),
                None,
                Some(vec![1, 2]),
                None,
                Some(vec![3, 4, 5]),
                Some(vec![3, 4, 5]),
            ],
        );

        // The values of the built array are the ones its elements reach, and no more: the ones a
        // null element would have covered are never appended.
        assert_eq!(built.values().len(), 13);
    }

    #[test]
    fn gathering() {
        let array = array();

        let mut builder = builder();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 9], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(vec![3, 4, 5]),
                Some(vec![1, 2]),
                None,
                Some(vec![1, 2]),
                None,
            ],
        );
    }

    #[test]
    fn gathering_consecutive_indices_is_one_subslice() {
        let array = array();

        let mut builder = builder();
        unsafe { builder.gather_extend(&array, &[0, 1, 2, 0], ShareStrategy::Always) };

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(vec![1, 2]),
                None,
                Some(vec![3, 4, 5]),
                Some(vec![1, 2]),
            ],
        );
    }

    #[test]
    fn a_scalar_array_is_appended_without_being_materialized() {
        let array = PlListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );

        let mut builder = PlListArrayBuilder::new(builder_like(array.values()));
        builder.subslice_extend(&array, 999_999_998, 2, ShareStrategy::Always);
        builder.subslice_extend_each_repeated(&array, 0, 1, 2, ShareStrategy::Always);
        unsafe { builder.gather_extend(&array, &[999_999_999], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[0, 1_000_000_000], ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                Some(vec![1, 2]),
                None,
            ],
        );
    }

    #[test]
    fn a_fully_null_array_appends_no_values() {
        let array = PlListArray::new_full_null(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );

        let mut builder = PlListArrayBuilder::new(builder_like(array.values()));
        builder.subslice_extend(&array, 0, 3, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 3);
        assert_eq!(built.null_count(), 3);
    }

    #[test]
    fn freeze_reset_leaves_an_empty_builder() {
        let array = array();

        let mut builder = builder();
        builder.extend(&array, ShareStrategy::Always);
        assert_eq!(builder.freeze_reset().len(), 3);

        assert!(builder.is_empty());
        assert!(builder.values().is_empty());
        builder.extend_nulls(1);
        assert_eq!(builder.freeze().len(), 1);
    }

    #[test]
    fn a_builder_over_a_typed_child_needs_no_trait_object() {
        use crate::PlPrimitiveArrayBuilder;

        let array = array();
        let mut builder = PlListArrayBuilder::new(PlPrimitiveArrayBuilder::<i32>::new());
        builder.extend(&array, ShareStrategy::Always);

        assert_eq!(elements(&builder.freeze()), elements(&array));
    }
}
