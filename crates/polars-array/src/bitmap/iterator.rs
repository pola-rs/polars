use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the bits of a [`PlBitmap`](super::PlBitmap) or a [`PlBitmapRef`].
///
/// A broadcast mask is iterated without being materialized, so this is `O(1)` in memory regardless
/// of the mask's length.
#[derive(Clone)]
pub struct PlBitmapIter<'a> {
    mask: PlBitmapRef<'a>,
    range: Range<usize>,
}

impl<'a> PlBitmapIter<'a> {
    #[inline]
    pub(super) fn new(mask: PlBitmapRef<'a>) -> Self {
        Self {
            range: 0..mask.len(),
            mask,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> bool {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the mask.
        unsafe { self.mask.get_unchecked(i) }
    }
}

impl Iterator for PlBitmapIter<'_> {
    type Item = bool;

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

impl DoubleEndedIterator for PlBitmapIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlBitmapIter<'_> {}
unsafe impl TrustedLen for PlBitmapIter<'_> {}
