//! The builder of a [`PlBinaryArray`].

use polars_arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;

use super::PlBinaryArray;
use crate::bitmap::PlBitmap;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, for_each_run, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBinaryArray`].
pub struct PlBinaryArrayBuilder {
    values: Vec<u8>,
    /// The start of every element appended so far, plus the end of the last.
    offsets: Vec<u64>,
    validity: OptBitmapBuilder,
}

impl PlBinaryArrayBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0],
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        StaticArrayBuilder::reserve(&mut builder, capacity);
        builder
    }

    /// The end of the last element appended, which is where the next one starts.
    #[inline]
    fn last_offset(&self) -> u64 {
        self.offsets[self.offsets.len() - 1]
    }

    /// Appends one element covering the `length` bytes that were just appended.
    #[inline]
    fn push_offset(&mut self, length: usize) {
        self.offsets.push(self.last_offset() + length as u64);
    }

    /// Appends `value` as an element of its own.
    #[inline]
    pub fn push_value(&mut self, value: &[u8]) {
        self.values.extend_from_slice(value);
        self.push_offset(value.len());
        self.validity.extend_constant(1, true);
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        self.push_offset(0);
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    #[inline]
    pub fn push(&mut self, value: Option<&[u8]>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Appends `element` `repeats` times over, one element per copy.
    fn extend_repeated(&mut self, element: &[u8], repeats: usize) {
        self.values.reserve(repeats * element.len());
        self.offsets.reserve(repeats);

        let mut offset = self.last_offset();
        for _ in 0..repeats {
            self.values.extend_from_slice(element);
            offset += element.len() as u64;
            self.offsets.push(offset);
        }
    }

    /// Appends the `length` elements of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(&mut self, other: &PlBinaryArray, start: usize, length: usize) {
        let Some(offsets) = other.flat_offsets() else {
            if let Some(element) = other.scalar_value_ignore_validity() {
                self.extend_repeated(element, length);
            }
            return;
        };

        let (first, last) = (offsets[start], offsets[start + length]);
        // SAFETY: the offsets are ordered and bounded by the length of the values of `other`.
        let bytes = unsafe { other.values().get_unchecked(first as usize..last as usize) };
        self.values.extend_from_slice(bytes);

        let base = self.last_offset();
        self.offsets.extend(
            offsets[start + 1..=start + length]
                .iter()
                .map(|offset| base + (offset - first)),
        );
    }
}

impl Default for PlBinaryArrayBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StaticArrayBuilder for PlBinaryArrayBuilder {
    type Array = PlBinaryArray;

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn freeze(self) -> PlBinaryArray {
        let length = self.offsets.len() - 1;
        let validity = self.validity.into_opt_validity().map(PlBitmap::from_bitmap);
        // SAFETY: the offsets hold the start of every element plus the end of the last, they are
        // pushed in non-decreasing order, and they reach exactly the bytes appended alongside them.
        unsafe {
            PlBinaryArray::new_unchecked(
                Buffer::from(self.values),
                Buffer::from(self.offsets),
                length,
                validity,
            )
        }
    }

    fn freeze_reset(&mut self) -> PlBinaryArray {
        let values = std::mem::take(&mut self.values);
        let offsets = std::mem::replace(&mut self.offsets, vec![0]);
        let validity = std::mem::take(&mut self.validity);
        let length = offsets.len() - 1;
        // SAFETY: as in `freeze`.
        unsafe {
            PlBinaryArray::new_unchecked(
                Buffer::from(values),
                Buffer::from(offsets),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        let offset = self.last_offset();
        self.offsets.extend(std::iter::repeat_n(offset, length));
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlBinaryArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length);

        self.extend_values(other, start, length);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlBinaryArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.offsets.reserve(length * repeats);

        if other.offsets_are_flat() {
            for i in start..start + length {
                // SAFETY: `i` is in bounds of the array, whose offsets are flat.
                let element = unsafe { other.value_unchecked(i) };
                self.extend_repeated(element, repeats);
            }
        } else {
            self.extend_values(other, start, length * repeats);
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
        other: &PlBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.offsets.reserve(idxs.len());

        if other.offsets_are_flat() {
            for_each_run(idxs, |first, run_length| {
                self.extend_values(other, first, run_length);
            });
        } else {
            self.extend_values(other, 0, idxs.len());
        }

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlBinaryArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.offsets.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                let element = unsafe { other.value_unchecked(idx) };
                self.values.extend_from_slice(element);
                self.push_offset(element.len());
            } else {
                self.push_offset(0);
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}
