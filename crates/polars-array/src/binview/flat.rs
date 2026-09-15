//! What a [`PlBinaryViewArray`] gains from being known to be [`Flat`].

use arrow::array::View;
use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::{PlBinaryViewArray, PlBinaryViewIter};
use crate::flat::Flat;

/// The methods a [`PlBinaryViewArray`] gains from holding one view and one mask bit per element.
impl Flat<PlBinaryViewArray> {
    /// The backing views buffer, holding exactly [`len`](PlBinaryViewArray::len) slots.
    #[inline(always)]
    pub const fn views(&self) -> &Buffer<View> {
        &self.as_array().views
    }

    /// Returns the view of the element at `i`.
    #[inline]
    pub fn view(&self, i: usize) -> View {
        assert!(i < self.as_array().length, "index out of bounds");
        unsafe { self.view_unchecked(i) }
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn view_unchecked(&self, i: usize) -> View {
        debug_assert!(i < self.as_array().length);
        unsafe { *self.as_array().views.get_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.as_array().length);
        // SAFETY: every view reads bytes the data buffers hold, upheld by every constructor.
        unsafe {
            self.as_array()
                .views
                .get_unchecked(i)
                .get_slice_unchecked(self.as_array().buffers.as_slice())
        }
    }

    /// Consumes this array into its views, the data buffers they read, and its validity mask.
    #[inline]
    pub fn into_inner(self) -> (Buffer<View>, Buffer<Buffer<u8>>, Option<Bitmap>) {
        let PlBinaryViewArray {
            views,
            buffers,
            length: _,
            validity,
        } = self.into_array();

        (views, buffers, validity)
    }
}

crate::impl_flat_methods!(PlBinaryViewArray, &[u8]);

crate::impl_into_iterator!(Flat<PlBinaryViewArray>, PlBinaryViewIter<'a>);

/// Compares an array of unknown representation against a flat one.
impl PartialEq<Flat<PlBinaryViewArray>> for PlBinaryViewArray {
    #[inline]
    fn eq(&self, other: &Flat<PlBinaryViewArray>) -> bool {
        *self == *other.as_array()
    }
}
