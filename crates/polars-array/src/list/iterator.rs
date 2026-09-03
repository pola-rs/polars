use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use super::PlListArray;
use crate::array::PlArray;
use crate::broadcast::is_broadcastable;

/// Iterator over the elements of a [`PlListArray`], ignoring validity.
#[derive(Clone)]
pub struct PlListValuesIter<'a> {
    array: &'a PlListArray,
    range: Range<usize>,
}

impl<'a> PlListValuesIter<'a> {
    #[inline]
    pub(super) fn new(array: &'a PlListArray) -> Self {
        Self::new_broadcast(array, array.len())
    }

    /// # Safety
    /// `array` must broadcast to `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new_broadcast(array: &'a PlListArray, length: usize) -> Self {
        debug_assert!(is_broadcastable(array.len(), length));

        Self {
            array,
            range: 0..length,
        }
    }

    /// Whether the array holds the one element every position reads, rather than one each.
    #[inline(always)]
    fn is_broadcast(&self) -> bool {
        self.array.len() == 1
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Box<dyn PlArray> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array unless the array
        // is being broadcast, in which case it holds the one element every position reads.
        unsafe {
            self.array
                .value_unchecked(if self.is_broadcast() { 0 } else { i })
        }
    }
}

impl Iterator for PlListValuesIter<'_> {
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

    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for PlListValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth_back(n).map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlListValuesIter<'_> {}
unsafe impl TrustedLen for PlListValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlListArray`].
#[derive(Clone)]
pub struct PlListIter<'a> {
    array: &'a PlListArray,
    range: Range<usize>,
}

impl<'a> PlListIter<'a> {
    #[inline]
    pub(super) fn new(array: &'a PlListArray) -> Self {
        Self::new_broadcast(array, array.len())
    }

    /// # Safety
    /// `array` must broadcast to `length`, per [`crate::broadcast`].
    #[inline]
    pub(super) fn new_broadcast(array: &'a PlListArray, length: usize) -> Self {
        debug_assert!(is_broadcastable(array.len(), length));

        Self {
            array,
            range: 0..length,
        }
    }

    /// Whether the array holds the one element every position reads, rather than one each.
    #[inline(always)]
    fn is_broadcast(&self) -> bool {
        self.array.len() == 1
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<Box<dyn PlArray>> {
        // SAFETY: `i` comes from `self.range`, so it is in bounds of the array unless the array
        // is being broadcast, in which case it holds the one element every position reads.
        unsafe {
            self.array
                .get_unchecked(if self.is_broadcast() { 0 } else { i })
        }
    }
}

impl Iterator for PlListIter<'_> {
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

    #[inline]
    fn count(self) -> usize {
        self.range.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for PlListIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|i| self.get(i))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth_back(n).map(|i| self.get(i))
    }
}

impl ExactSizeIterator for PlListIter<'_> {}
unsafe impl TrustedLen for PlListIter<'_> {}

#[cfg(test)]
mod tests {

    use polars_buffer::Buffer;

    use crate::iterator_tests::assert_iterates;
    use crate::{PlArray, PlListArray, PlPrimitiveArray};

    /// The list `values` are, as an element of a list array is.
    fn element(values: &[i32]) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.to_vec()))
    }

    /// A flat array of the lists `[1, 2]`, `[]` and `[3]`.
    fn flat_array() -> PlListArray {
        PlListArray::new(
            element(&[1, 2, 3]),
            Buffer::from_owner([0, 2, 2, 3]),
            3,
            None,
        )
    }

    fn elements() -> [Box<dyn PlArray>; 3] {
        [element(&[1, 2]), element(&[]), element(&[3])]
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlListArray::new_scalar(element(&[1, 2]), 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some(element(&[1, 2])));
        assert_eq!(array.iter().last(), Some(Some(element(&[1, 2]))));
    }
}
