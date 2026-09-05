//! The builder of a [`PlListArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlListArray;
use crate::bitmap::PlBitmap;
use crate::broadcast::ArrayRepr;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlListArray`].
pub struct PlListArrayBuilder<B: PlArrayBuilder = Box<dyn PlArrayBuilder>> {
    /// The start of every element appended so far, plus the end of the last: one slot more than the
    /// elements, which is what the offsets of a flat list array hold.
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

    /// The builder of the values the lists are taken over, so that the values one element covers
    /// can be appended to it directly.
    ///
    /// Everything appended through here becomes part of the element that the next
    /// [`finish_row`](Self::finish_row) closes. Until one does, the values are past the end of
    /// every element the builder holds, and are dropped by a
    /// [`freeze`](StaticArrayBuilder::freeze) that never closes them.
    #[inline]
    pub fn values_mut(&mut self) -> &mut B {
        &mut self.values
    }

    /// Closes one element, covering every value appended to the child since the last element was.
    #[inline]
    pub fn finish_row(&mut self) {
        // Every element ends where the child ended when it was closed, so the values appended
        // since then are exactly the ones past the end of the last element.
        let end = self.values.len() as u64;
        debug_assert!(end >= self.last_offset(), "the child builder cannot shrink");
        self.offsets.push(end);
        self.validity.extend_constant(1, true);
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
    fn extend_values(
        &mut self,
        other: &PlListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let offsets = match other.offsets_repr() {
            // Every element covers the same range, which is appended once per element.
            ArrayRepr::Scalar(range) => {
                let width = range.end - range.start;
                self.values.subslice_extend_repeated(
                    other.values(),
                    range.start as usize,
                    width as usize,
                    length,
                    share,
                );

                let mut offset = self.last_offset();
                self.offsets.reserve(length);
                for _ in 0..length {
                    offset += width;
                    self.offsets.push(offset);
                }
                return;
            },
            ArrayRepr::Flat(offsets) => offsets,
        };

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
        let validity = self.validity.into_opt_validity().map(PlBitmap::from_bitmap);
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
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
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

        if other.offsets_are_flat() {
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
        } else {
            // Every element covers the same range, so which of them is repeated is immaterial.
            self.extend_values(other, start, length * repeats, share);
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

        if other.offsets_are_flat() {
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
        } else {
            // Every index reads the one range the array holds.
            self.extend_values(other, 0, idxs.len(), share);
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
    use crate::bitmap::PlBitmap;
    use crate::builder::{ShareStrategy, StaticArrayBuilder, builder_like};
    use crate::{PlListArray, PlPrimitiveArray};

    /// Three lists over five values: `[1, 2]`, null and `[3, 4, 5]`.
    fn array() -> PlListArray {
        PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
            Buffer::from(vec![0u64, 2, 2, 5]),
        )
        .with_validity(Some(PlBitmap::from_bitmap(Bitmap::from_iter([
            true, false, true,
        ]))))
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
    fn appending_one_element_at_a_time() {
        let values = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let mut builder = builder();
        // Two values appended to the child, closed as one element covering both.
        builder
            .values_mut()
            .subslice_extend(&values, 0, 2, ShareStrategy::Always);
        builder.finish_row();
        // No values at all, closed as the empty list.
        builder.finish_row();
        builder.extend_nulls(1);
        builder
            .values_mut()
            .subslice_extend(&values, 2, 1, ShareStrategy::Always);
        builder.finish_row();

        let built = builder.freeze();
        assert_eq!(
            elements(&built),
            [Some(vec![1, 2]), Some(vec![]), None, Some(vec![3])],
        );
    }

    #[test]
    fn values_appended_without_being_closed_are_dropped() {
        let values = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let mut builder = builder();
        builder
            .values_mut()
            .subslice_extend(&values, 0, 1, ShareStrategy::Always);
        builder.finish_row();
        // Appended but never closed: no element covers them.
        builder
            .values_mut()
            .subslice_extend(&values, 1, 2, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(elements(&built), [Some(vec![1])]);
        // The child still holds them; the offsets are what say they are past the last element.
        assert_eq!(built.values().len(), 3);
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
}
