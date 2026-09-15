//! The builder of a [`PlListArray`].

use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlListArray;
use crate::bitmap::PlBitmap;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, for_each_run,
    gather_extend_validity, opt_gather_extend_validity, subslice_extend_each_repeated_validity,
    subslice_extend_validity,
};

/// A builder of a [`PlListArray`].
pub struct PlListArrayBuilder<B: PlArrayBuilder = Box<dyn PlArrayBuilder>> {
    /// The start of every element appended so far, plus the end of the last.
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

    /// The builder of the values the lists are taken over, appended to directly.
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
        // Every element of a scalar array covers the same range, which is appended once per
        // element.
        if let Some(range) = other.scalar_offsets() {
            let width = range.end - range.start;
            self.values
                .subslice_extend_repeated(other.values(), range.start, width, length, share);

            let mut offset = self.last_offset();
            self.offsets.reserve(length);
            for _ in 0..length {
                offset += width as u64;
                self.offsets.push(offset);
            }
            return;
        }

        let offsets = other.flat_offsets().unwrap();

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
            for_each_run(idxs, |first, run_length| {
                self.extend_values(other, first, run_length, share);
            });
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
