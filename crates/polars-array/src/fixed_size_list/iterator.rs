use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use super::PlFixedSizeListArray;
use crate::array::PlArray;
use crate::broadcast::{broadcast_index, is_broadcastable};

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
        Self::new_broadcast(array, array.len())
    }

    /// # Safety
    /// `array` must broadcast to `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new_broadcast(array: &'a PlFixedSizeListArray, length: usize) -> Self {
        debug_assert!(is_broadcastable(array.len(), length));

        Self {
            range: 0..length,
            array,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Box<dyn PlArray> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array unless the array
        // is being broadcast, in which case it holds the one element every position reads.
        unsafe {
            self.array
                .value_unchecked(broadcast_index(i, self.array.len()))
        }
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
        Self::new_broadcast(array, array.len())
    }

    /// # Safety
    /// `array` must broadcast to `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new_broadcast(array: &'a PlFixedSizeListArray, length: usize) -> Self {
        debug_assert!(is_broadcastable(array.len(), length));

        Self {
            range: 0..length,
            array,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Box<dyn PlArray>> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array unless the array
        // is being broadcast, in which case it holds the one element every position reads.
        unsafe {
            self.array
                .get_unchecked(broadcast_index(i, self.array.len()))
        }
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
