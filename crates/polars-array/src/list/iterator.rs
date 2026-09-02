use std::ops::Range;

use arrow::trusted_len::TrustedLen;

use super::PlListArray;
use crate::array::PlArray;
use crate::broadcast::is_broadcastable;

/// Iterator over the elements of a [`PlListArray`], ignoring validity.
///
/// Each element is the values array sliced to the range that element covers, which is `O(1)`.
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
    ///
    /// A single element is what an array being broadcast holds — and what an array of one element
    /// holds when it is not. Either way position zero is the only element there is. This does not
    /// depend on the position being read, so a loop over them settles it once rather than at every
    /// one.
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
///
/// A scalar validity mask is not materialized, and neither is the values array of an element.
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
    ///
    /// A single element is what an array being broadcast holds — and what an array of one element
    /// holds when it is not. Either way position zero is the only element there is. This does not
    /// depend on the position being read, so a loop over them settles it once rather than at every
    /// one.
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
    use arrow::bitmap::Bitmap;
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
    fn flat_under_a_flat_mask() {
        let array = flat_array().with_validity(Some(Bitmap::from_iter([true, false, true])));
        let [first, _, last] = elements();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &[Some(first), None, Some(last)]);
    }

    #[test]
    fn flat_under_a_scalar_mask() {
        let array = flat_array().with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        assert_iterates(array.iter(), &[None, None, None]);
    }

    #[test]
    fn scalar() {
        let array = PlListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    #[test]
    fn scalar_under_a_flat_mask() {
        let array = PlListArray::new_scalar(element(&[1, 2]), 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert_iterates(
            array.iter(),
            &[Some(element(&[1, 2])), None, Some(element(&[1, 2]))],
        );
    }

    #[test]
    fn empty() {
        let array = PlListArray::new_empty(element(&[]));

        assert_iterates(array.values_iter(), &[]);
        assert_iterates(array.iter(), &[]);
    }

    #[test]
    fn broadcast() {
        let array = PlListArray::new(element(&[1, 2]), Buffer::from_owner([0, 2]), 1, None);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.broadcast_values_iter(4), &expected);
        assert_iterates(array.broadcast_iter(4), &expected.map(Some));

        // An array is its own broadcast to the length it already has.
        assert_iterates(flat_array().broadcast_iter(3), &elements().map(Some));
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
