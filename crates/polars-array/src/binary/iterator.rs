use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;
use crate::broadcast::broadcast_index;

/// Iterator over the elements of a [`PlBinaryArray`](super::PlBinaryArray), ignoring validity.
///
/// Each element is the values buffer sliced to the range that element covers, which is `O(1)`.
#[derive(Clone)]
pub struct PlBinaryValuesIter<'a> {
    values: &'a [u8],
    offsets: &'a [u64],
    range: Range<usize>,
}

impl<'a> PlBinaryValuesIter<'a> {
    /// # Safety
    /// `offsets` must be flat (`length + 1` offsets) or scalar (two offsets) for `length`, per
    /// [`crate::broadcast`], and must be ordered and bounded by the length of `values`.
    #[inline]
    pub(super) fn new(values: &'a [u8], offsets: &'a [u64], length: usize) -> Self {
        Self {
            values,
            offsets,
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> &'a [u8] {
        // Scalar offsets hold the one range every element covers, so they are read at slot zero.
        let i = broadcast_index(i, self.offsets.len() - 1);

        // SAFETY: `i` comes from `self.range`, so the range it reads is in bounds of the offsets:
        // they hold one slot more than the starts `broadcast_index` maps onto. The offsets are
        // ordered and bounded by the length of the values, so the range is in bounds of those.
        unsafe {
            let start = *self.offsets.get_unchecked(i) as usize;
            let end = *self.offsets.get_unchecked(i + 1) as usize;
            self.values.get_unchecked(start..end)
        }
    }
}

impl<'a> Iterator for PlBinaryValuesIter<'a> {
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

impl DoubleEndedIterator for PlBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlBinaryValuesIter<'_> {}
unsafe impl TrustedLen for PlBinaryValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlBinaryArray`](super::PlBinaryArray).
///
/// Neither a scalar validity mask nor scalar offsets are materialized.
#[derive(Clone)]
pub struct PlBinaryIter<'a> {
    values: PlBinaryValuesIter<'a>,
    validity: Option<PlBitmapRef<'a>>,
}

impl<'a> PlBinaryIter<'a> {
    /// # Safety
    /// `offsets` must be flat or scalar for `length`, per [`crate::broadcast`], and `validity`
    /// must have `length` bits.
    #[inline]
    pub(super) fn new(
        values: &'a [u8],
        offsets: &'a [u64],
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        Self {
            values: PlBinaryValuesIter::new(values, offsets, length),
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

impl<'a> Iterator for PlBinaryIter<'a> {
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

impl DoubleEndedIterator for PlBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.values.range.end.checked_sub(1)?;
        let value = self.values.next_back()?;
        Some(self.is_valid(i).then_some(value))
    }
}

impl ExactSizeIterator for PlBinaryIter<'_> {}
unsafe impl TrustedLen for PlBinaryIter<'_> {}
