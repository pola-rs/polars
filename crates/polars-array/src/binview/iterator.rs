use std::ops::Range;

use arrow::array::View;
use arrow::trusted_len::TrustedLen;
use polars_buffer::Buffer;

use crate::bitmap::PlBitmapRef;
use crate::broadcast::broadcast_index;

/// Iterator over the values of a [`PlBinaryViewArray`](super::PlBinaryViewArray), ignoring
/// validity.
#[derive(Clone)]
pub struct PlBinaryViewValuesIter<'a> {
    views: &'a [View],
    buffers: &'a [Buffer<u8>],
    range: Range<usize>,
}

impl<'a> PlBinaryViewValuesIter<'a> {
    /// # Safety
    /// `views` must be flat or scalar for `length`, per [`crate::broadcast`], and every one of
    /// them must read bytes that `buffers` holds.
    #[inline]
    pub(super) fn new(views: &'a [View], buffers: &'a [Buffer<u8>], length: usize) -> Self {
        Self {
            views,
            buffers,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> &'a [u8] {
        // SAFETY: `i` comes from `self.range`, so the scalar index is in bounds, and the view
        // there reads bytes the buffers hold.
        unsafe {
            self.views
                .get_unchecked(broadcast_index(i, self.views.len()))
                .get_slice_unchecked(self.buffers)
        }
    }
}

impl<'a> Iterator for PlBinaryViewValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| self.get(i))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(|i| self.get(i))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl DoubleEndedIterator for PlBinaryViewValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlBinaryViewValuesIter<'_> {}
unsafe impl TrustedLen for PlBinaryViewValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlBinaryViewArray`](super::PlBinaryViewArray).
#[derive(Clone)]
pub struct PlBinaryViewIter<'a> {
    values: PlBinaryViewValuesIter<'a>,
    validity: Option<PlBitmapRef<'a>>,
}

impl<'a> PlBinaryViewIter<'a> {
    /// # Safety
    /// `views` must be flat or scalar for `length`, per [`crate::broadcast`], every one of them
    /// must read bytes that `buffers` holds, and `validity` must have `length` bits.
    #[inline]
    pub(super) fn new(
        views: &'a [View],
        buffers: &'a [Buffer<u8>],
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        Self {
            values: PlBinaryViewValuesIter::new(views, buffers, length),
            validity,
        }
    }

    /// Whether the element the values iterator is about to yield at `i` is valid.
    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        // SAFETY: `i` is a position the values iterator has left to yield, so it is in bounds of
        // the mask, which has one bit per element.
        self.validity
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
    }
}

impl<'a> Iterator for PlBinaryViewIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.values.range.start;
        let value = self.values.next()?;
        Some(self.is_valid(i).then_some(value))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.values.range.start.checked_add(n)?;
        let value = self.values.nth(n)?;
        Some(self.is_valid(i).then_some(value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl DoubleEndedIterator for PlBinaryViewIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.values.range.end.checked_sub(1)?;
        let value = self.values.next_back()?;
        Some(self.is_valid(i).then_some(value))
    }
}

impl ExactSizeIterator for PlBinaryViewIter<'_> {}
unsafe impl TrustedLen for PlBinaryViewIter<'_> {}
