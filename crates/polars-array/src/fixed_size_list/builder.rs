//! The builder of a [`PlFixedSizeListArray`].

use polars_arrow::bitmap::OptBitmapBuilder;
use polars_utils::IdxSize;

use super::PlFixedSizeListArray;
use crate::bitmap::PlBitmap;
use crate::builder::{
    PlArrayBuilder, ShareStrategy, StaticArrayBuilder, assert_subslice, for_each_run,
    gather_extend_validity, opt_gather_extend_validity, subslice_extend_each_repeated_validity,
    subslice_extend_validity,
};

/// A builder of a [`PlFixedSizeListArray`].
pub struct PlFixedSizeListArrayBuilder<B: PlArrayBuilder = Box<dyn PlArrayBuilder>> {
    values: B,
    width: usize,
    length: usize,
    validity: OptBitmapBuilder,
}

impl<B: PlArrayBuilder> PlFixedSizeListArrayBuilder<B> {
    /// Creates an empty builder of the lists of `width` values the child builder builds.
    pub fn new(values: B, width: usize) -> Self {
        Self {
            values,
            width,
            length: 0,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(values: B, width: usize, capacity: usize) -> Self {
        let mut builder = Self::new(values, width);
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The builder of the values the lists are taken over.
    #[inline]
    pub fn values(&self) -> &B {
        &self.values
    }

    /// The number of values every element of the built array covers.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The builder of the values the lists are taken over, appended to directly.
    #[inline]
    pub fn values_mut(&mut self) -> &mut B {
        &mut self.values
    }

    /// Closes one element, covering the width of values appended to the child since the last.
    #[inline]
    pub fn finish_row(&mut self) {
        assert_eq!(
            self.values.len(),
            (self.length + 1) * self.width,
            "an element of a fixed size list builder of width {} covers {} values, \
             but the child holds {} for the {} elements before it",
            self.width,
            self.width,
            self.values.len(),
            self.length,
        );
        self.length += 1;
        self.validity.push(true);
    }

    /// Panics unless `other` is as wide as the lists this builds.
    fn assert_width(&self, other: &PlFixedSizeListArray) {
        assert_eq!(
            other.width(),
            self.width,
            "cannot append a fixed size list array of width {} to a builder of width {}",
            other.width(),
            self.width,
        );
    }

    /// Appends the `length` elements of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        if let Some(values) = other.flat_values() {
            self.values
                .subslice_extend(values, start * self.width, length * self.width, share);
        } else if let Some(element) = other.scalar_value_ignore_validity() {
            self.values
                .subslice_extend_repeated(element, 0, self.width, length, share);
        }
    }
}

impl<B: PlArrayBuilder> StaticArrayBuilder for PlFixedSizeListArrayBuilder<B> {
    type Array = PlFixedSizeListArray;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.width);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    fn freeze(self) -> PlFixedSizeListArray {
        let (width, length) = (self.width, self.length);
        let validity = self.validity.into_opt_validity().map(PlBitmap::from_bitmap);
        // SAFETY: every element appended the width of values it covers to the child, so the values
        // hold the width of every element laid end to end, and the mask holds one bit per element.
        unsafe {
            PlFixedSizeListArray::new_unchecked(self.values.freeze(), width, length, validity)
        }
    }

    fn freeze_reset(&mut self) -> PlFixedSizeListArray {
        let validity = std::mem::take(&mut self.validity);
        let length = std::mem::take(&mut self.length);
        // SAFETY: as in `freeze`.
        unsafe {
            PlFixedSizeListArray::new_unchecked(
                self.values.freeze_reset(),
                self.width,
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        self.values.extend_nulls(length * self.width);
        self.validity.extend_constant(length, false);
        self.length += length;
    }

    fn subslice_extend(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);

        self.extend_values(other, start, length, share);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
        self.length += length;
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlFixedSizeListArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        assert_subslice(other.len(), start, length);
        self.values.reserve(length * repeats * self.width);

        if let Some(values) = other.flat_values() {
            for i in start..start + length {
                self.values.subslice_extend_repeated(
                    values,
                    i * self.width,
                    self.width,
                    repeats,
                    share,
                );
            }
        } else {
            self.extend_values(other, start, length * repeats, share);
        }

        subslice_extend_each_repeated_validity(
            &mut self.validity,
            other.validity(),
            start,
            length,
            repeats,
        );
        self.length += length * repeats;
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PlFixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        if other.values_are_flat() {
            for_each_run(idxs, |first, run_length| {
                self.extend_values(other, first, run_length, share);
            });
        } else {
            self.extend_values(other, 0, idxs.len(), share);
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
        self.length += idxs.len();
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlFixedSizeListArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.assert_width(other);
        self.values.reserve(idxs.len() * self.width);

        let values = other.values();

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                let range = unsafe { other.value_range_unchecked(idx) };
                self.values
                    .subslice_extend(values, range.start, self.width, share);
            } else {
                self.values.extend_nulls(self.width);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
        self.length += idxs.len();
    }
}
