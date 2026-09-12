//! What a [`PlFixedSizeBinaryArray`] gains from being known to be [`Flat`].

use polars_buffer::Buffer;

use super::PlFixedSizeBinaryArray;
use crate::flat::Flat;

/// The methods a [`PlFixedSizeBinaryArray`] gains from holding one slot and one bit per element.
impl Flat<PlFixedSizeBinaryArray> {
    /// The backing values buffer, holding `len * width` bytes.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.as_array().values
    }

    /// The values as a slice of `len * width` bytes.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        self.as_array().values.as_slice()
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.as_array().length);
        let start = i * self.as_array().width;
        // SAFETY: the values hold the width of every element, so the element at `i` is in bounds.
        unsafe {
            self.as_array()
                .values
                .get_unchecked(start..start + self.as_array().width)
        }
    }
}

crate::impl_flat_methods!(PlFixedSizeBinaryArray, &[u8]);
