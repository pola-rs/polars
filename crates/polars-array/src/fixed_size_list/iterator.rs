use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use super::PlFixedSizeListArray;
use crate::array::PlArray;

/// Iterator over the elements of a [`PlFixedSizeListArray`], ignoring validity.
///
/// Each element is the values array sliced to the range that element covers, which is `O(1)`.
#[derive(Clone)]
pub struct PlFixedSizeListValuesIter<'a> {
    array: &'a PlFixedSizeListArray,
    range: Range<usize>,
}

impl<'a> PlFixedSizeListValuesIter<'a> {
    #[inline]
    pub(super) fn new(array: &'a PlFixedSizeListArray) -> Self {
        Self {
            range: 0..array.len(),
            array,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Box<dyn PlArray> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array.
        unsafe { self.array.value_unchecked(i) }
    }
}

impl Iterator for PlFixedSizeListValuesIter<'_> {
    type Item = Box<dyn PlArray>;

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

impl DoubleEndedIterator for PlFixedSizeListValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlFixedSizeListValuesIter<'_> {}
unsafe impl TrustedLen for PlFixedSizeListValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlFixedSizeListArray`].
///
/// A scalar validity mask is not materialized, and neither is the values array of an element.
#[derive(Clone)]
pub struct PlFixedSizeListIter<'a> {
    array: &'a PlFixedSizeListArray,
    range: Range<usize>,
}

impl<'a> PlFixedSizeListIter<'a> {
    #[inline]
    pub(super) fn new(array: &'a PlFixedSizeListArray) -> Self {
        Self {
            range: 0..array.len(),
            array,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Box<dyn PlArray>> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array.
        unsafe { self.array.get_unchecked(i) }
    }
}

impl Iterator for PlFixedSizeListIter<'_> {
    type Item = Option<Box<dyn PlArray>>;

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

impl DoubleEndedIterator for PlFixedSizeListIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlFixedSizeListIter<'_> {}
unsafe impl TrustedLen for PlFixedSizeListIter<'_> {}
