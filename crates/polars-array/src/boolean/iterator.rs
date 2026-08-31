use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the optional elements of a [`PlBooleanArray`](super::PlBooleanArray).
///
/// Neither a scalar values bitmap nor a scalar validity mask is materialized, so this is
/// `O(1)` in memory regardless of the array's length.
#[derive(Clone)]
pub struct PlBooleanIter<'a> {
    values: PlBitmapRef<'a>,
    validity: Option<PlBitmapRef<'a>>,
    range: Range<usize>,
}

impl<'a> PlBooleanIter<'a> {
    /// # Safety
    /// `values` and `validity` must both have `length` bits.
    #[inline]
    pub(super) fn new(
        values: PlBitmapRef<'a>,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        debug_assert_eq!(values.len(), length);
        debug_assert!(validity.is_none_or(|v| v.len() == length));

        Self {
            values,
            validity,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<bool> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of both masks.
        let is_valid = self
            .validity
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) });

        is_valid.then(|| unsafe { self.values.get_unchecked(i) })
    }
}

impl Iterator for PlBooleanIter<'_> {
    type Item = Option<bool>;

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

impl DoubleEndedIterator for PlBooleanIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlBooleanIter<'_> {}
unsafe impl TrustedLen for PlBooleanIter<'_> {}
