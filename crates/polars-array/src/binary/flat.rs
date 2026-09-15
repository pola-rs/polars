//! What a [`PlBinaryArray`] gains from being known to be [`Flat`].

use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::PlBinaryArray;
use crate::flat::Flat;

/// The methods a [`PlBinaryArray`] gains from holding one range and one validity bit per element.
impl Flat<PlBinaryArray> {
    /// The backing offsets buffer, holding exactly [`len`](PlBinaryArray::len) `+ 1` offsets.
    #[inline(always)]
    pub const fn offsets(&self) -> &Buffer<u64> {
        &self.as_array().offsets
    }

    /// The backing values buffer, holding the bytes the offsets cut the elements out of.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.as_array().values
    }

    /// The values as a slice, which the offsets cut the elements out of.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.as_array().values.as_slice()
    }

    /// Consumes this array into its internal components, whose ranges and bits are one per element.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, Buffer<u64>, Option<Bitmap>) {
        let PlBinaryArray {
            values,
            offsets,
            length: _,
            validity,
        } = self.into_array();

        (values, offsets, validity)
    }

    /// The range of [`Self::values`] the element at `i` covers.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.as_array().length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.as_array().length);
        // SAFETY: the offsets hold one slot more than the starts, so `i + 1` is in bounds, and
        // every offset is at most the length of the values and therefore fits in a `usize`.
        unsafe {
            let start = *self.as_array().offsets.get_unchecked(i) as usize;
            let end = *self.as_array().offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the offsets are ordered and bounded by the length of the values.
        unsafe { self.as_array().values.get_unchecked(range) }
    }
}

crate::impl_flat_methods!(PlBinaryArray, &[u8]);
