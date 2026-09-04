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
    /// `usize::MAX` while the array holds one element per position, and `0` once it holds the one
    /// element every position reads.
    ///
    /// Resolving this once keeps the length of the array out of the loop, and folding a position
    /// through it costs no branch at all.
    index_mask: usize,
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
            // An array of a single element holds the one every position reads; one of as many
            // elements as there are positions holds one each.
            index_mask: if array.len() == 1 { 0 } else { usize::MAX },
        }
    }

    /// The element at position `i`, which the mask folds onto the one a broadcast array holds.
    ///
    /// # Safety
    /// `i` must be one of the positions the iterator was built for.
    #[inline(always)]
    unsafe fn get(array: &'a PlListArray, index_mask: usize, i: usize) -> Box<dyn PlArray> {
        // SAFETY: `i` is one of the iterator's positions, so it is in bounds of the array unless
        // the array is being broadcast, in which case the mask folds it onto the one element it
        // holds.
        unsafe { array.value_unchecked(i & index_mask) }
    }
}

impl Iterator for PlListValuesIter<'_> {
    type Item = Box<dyn PlArray>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.range.nth(n)?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
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

    /// Hoists the walk over the positions out of the loop, which `collect` and `for_each` route
    /// through: the positions are folded as the range they are, rather than stepped one `Option`
    /// at a time, and the array they read is loaded once.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (array, index_mask) = (self.array, self.index_mask);

        self.range.fold(init, |acc, i| {
            // SAFETY: the position comes from the range the iterator was built for.
            f(acc, unsafe { Self::get(array, index_mask, i) })
        })
    }
}

impl DoubleEndedIterator for PlListValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.range.next_back()?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.range.nth_back(n)?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    /// Hoists the walk over the positions out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (array, index_mask) = (self.array, self.index_mask);

        self.range.rfold(init, |acc, i| {
            // SAFETY: the position comes from the range the iterator was built for.
            f(acc, unsafe { Self::get(array, index_mask, i) })
        })
    }
}

impl ExactSizeIterator for PlListValuesIter<'_> {}
unsafe impl TrustedLen for PlListValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlListArray`].
#[derive(Clone)]
pub struct PlListIter<'a> {
    array: &'a PlListArray,
    range: Range<usize>,
    /// `usize::MAX` while the array holds one element per position, and `0` once it holds the one
    /// element every position reads.
    ///
    /// Resolving this once keeps the length of the array out of the loop, and folding a position
    /// through it costs no branch at all.
    index_mask: usize,
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
            // An array of a single element holds the one every position reads; one of as many
            // elements as there are positions holds one each.
            index_mask: if array.len() == 1 { 0 } else { usize::MAX },
        }
    }

    /// The element at position `i`, which the mask folds onto the one a broadcast array holds.
    ///
    /// # Safety
    /// `i` must be one of the positions the iterator was built for.
    #[inline(always)]
    unsafe fn get(array: &'a PlListArray, index_mask: usize, i: usize) -> Option<Box<dyn PlArray>> {
        // SAFETY: `i` is one of the iterator's positions, so it is in bounds of the array unless
        // the array is being broadcast, in which case the mask folds it onto the one element it
        // holds.
        unsafe { array.get_unchecked(i & index_mask) }
    }
}

impl Iterator for PlListIter<'_> {
    type Item = Option<Box<dyn PlArray>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.range.nth(n)?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
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

    /// Hoists the walk over the positions out of the loop, which `collect` and `for_each` route
    /// through: the positions are folded as the range they are, rather than stepped one `Option`
    /// at a time, and the array they read is loaded once.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (array, index_mask) = (self.array, self.index_mask);

        self.range.fold(init, |acc, i| {
            // SAFETY: the position comes from the range the iterator was built for.
            f(acc, unsafe { Self::get(array, index_mask, i) })
        })
    }
}

impl DoubleEndedIterator for PlListIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let i = self.range.next_back()?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let i = self.range.nth_back(n)?;
        // SAFETY: the position comes from the range the iterator was built for.
        Some(unsafe { Self::get(self.array, self.index_mask, i) })
    }

    /// Hoists the walk over the positions out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (array, index_mask) = (self.array, self.index_mask);

        self.range.rfold(init, |acc, i| {
            // SAFETY: the position comes from the range the iterator was built for.
            f(acc, unsafe { Self::get(array, index_mask, i) })
        })
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
        assert_eq!(
            array.values_iter().nth_back(999_999_999),
            Some(element(&[1, 2]))
        );
        assert_eq!(array.iter().last(), Some(Some(element(&[1, 2]))));
        assert_eq!(
            array.iter().nth_back(999_999_999),
            Some(Some(element(&[1, 2])))
        );
    }
}
