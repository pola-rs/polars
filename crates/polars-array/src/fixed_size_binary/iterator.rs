use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use crate::bitmap::PlBitmapRef;

/// Iterator over the elements of a [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray),
/// ignoring validity.
///
/// Each element is the values buffer sliced to the range that element covers, which is `O(1)`.
#[derive(Clone)]
pub struct PlFixedSizeBinaryValuesIter<'a> {
    values: &'a [u8],
    width: usize,
    /// How far apart the elements are: the width where the values are flat, and zero where they
    /// are scalar and every element covers the same bytes.
    stride: usize,
    range: Range<usize>,
}

impl<'a> PlFixedSizeBinaryValuesIter<'a> {
    /// # Safety
    /// `values` must be flat (`length * width` bytes) or scalar (`width` bytes) for `length`, per
    /// [`crate::broadcast`].
    #[inline]
    pub(super) fn new(values: &'a [u8], width: usize, length: usize) -> Self {
        Self {
            values,
            width,
            // Scalar values hold the one element every element covers, so every position reads
            // them from the start; flat ones lay the elements end to end, one width apart.
            stride: if values.len() == width { 0 } else { width },
            range: 0..length,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> &'a [u8] {
        let start = i * self.stride;
        // SAFETY: `i` comes from `self.range`, so the element it reads is in bounds of the values:
        // they are either flat over every position the range holds, or the one element all of them
        // read from the start.
        unsafe { self.values.get_unchecked(start..start + self.width) }
    }
}

impl<'a> Iterator for PlFixedSizeBinaryValuesIter<'a> {
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

impl DoubleEndedIterator for PlFixedSizeBinaryValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlFixedSizeBinaryValuesIter<'_> {}
unsafe impl TrustedLen for PlFixedSizeBinaryValuesIter<'_> {}

/// Iterator over the optional elements of a
/// [`PlFixedSizeBinaryArray`](super::PlFixedSizeBinaryArray).
///
/// Neither a scalar validity mask nor scalar values are materialized.
#[derive(Clone)]
pub struct PlFixedSizeBinaryIter<'a> {
    values: PlFixedSizeBinaryValuesIter<'a>,
    validity: Option<PlBitmapRef<'a>>,
}

impl<'a> PlFixedSizeBinaryIter<'a> {
    /// # Safety
    /// `values` must be flat or scalar for `length` and `width`, per [`crate::broadcast`], and
    /// `validity` must have `length` bits.
    #[inline]
    pub(super) fn new(
        values: &'a [u8],
        width: usize,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        Self {
            values: PlFixedSizeBinaryValuesIter::new(values, width, length),
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

impl<'a> Iterator for PlFixedSizeBinaryIter<'a> {
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

impl DoubleEndedIterator for PlFixedSizeBinaryIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.values.range.end.checked_sub(1)?;
        let value = self.values.next_back()?;
        Some(self.is_valid(i).then_some(value))
    }
}

impl ExactSizeIterator for PlFixedSizeBinaryIter<'_> {}
unsafe impl TrustedLen for PlFixedSizeBinaryIter<'_> {}
