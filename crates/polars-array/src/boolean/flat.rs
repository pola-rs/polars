//! What a [`PlBooleanArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::BitmapIter;

use super::{PlBooleanArray, PlBooleanIter};
use crate::flat::Flat;

/// The methods a [`PlBooleanArray`] gains from having one bit per element in every backing bitmap.
impl Flat<PlBooleanArray> {
    /// The values, as an ordinary [`Bitmap`] of exactly [`len`](PlBooleanArray::len) bits.
    #[inline(always)]
    pub const fn values(&self) -> &Bitmap {
        &self.as_array().values
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.as_array().length);
        unsafe { self.as_array().values.get_bit_unchecked(i) }
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> BitmapIter<'_> {
        self.as_array().values.iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBooleanIter<'_> {
        self.as_array().iter()
    }

    /// Consumes this array into its backing bitmaps, which both hold one bit per element.
    #[inline]
    pub fn into_inner(self) -> (Bitmap, Option<Bitmap>) {
        let PlBooleanArray {
            values,
            length: _,
            validity,
        } = self.into_array();

        (values, validity)
    }
}

crate::impl_flat_methods!(PlBooleanArray, bool);

crate::impl_into_iterator!(Flat<PlBooleanArray>, PlBooleanIter<'a>);

/// Compares an array of unknown representation against a flat one.
impl PartialEq<Flat<PlBooleanArray>> for PlBooleanArray {
    #[inline]
    fn eq(&self, other: &Flat<PlBooleanArray>) -> bool {
        *self == *other.as_array()
    }
}
