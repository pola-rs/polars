use arrow::array::View;
use arrow::trusted_len::TrustedLen;
use polars_buffer::Buffer;
use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::broadcast_slice;

/// Iterator over the values of a [`super::PlBinaryViewArray`], ignoring validity.
#[derive(Clone)]
pub struct PlBinaryViewValuesIter<'a> {
    views: SliceBroadcastIter<'a, View>,
    buffers: &'a [Buffer<u8>],
}

impl<'a> PlBinaryViewValuesIter<'a> {
    /// # Safety
    /// Every view must read bytes that `buffers` holds.
    #[inline]
    pub(super) fn new(views: &'a [View], buffers: &'a [Buffer<u8>], length: usize) -> Self {
        Self {
            views: broadcast_slice(views, length),
            buffers,
        }
    }

    /// The bytes `view` stands for.
    #[inline(always)]
    fn get(buffers: &'a [Buffer<u8>], view: &'a View) -> &'a [u8] {
        // SAFETY: the view is one of the array's, so it reads bytes the buffers hold.
        unsafe { view.get_slice_unchecked(buffers) }
    }
}

impl<'a> Iterator for PlBinaryViewValuesIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let view = self.views.next()?;
        Some(Self::get(self.buffers, view))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let view = self.views.nth(n)?;
        Some(Self::get(self.buffers, view))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.views.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.views.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        let view = self.views.last()?;
        Some(Self::get(self.buffers, view))
    }

    /// Hoists the representation out of the loop: flat views fold as the slice they are.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let buffers = self.buffers;
        self.views
            .fold(init, |acc, view| f(acc, Self::get(buffers, view)))
    }
}

impl DoubleEndedIterator for PlBinaryViewValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let view = self.views.next_back()?;
        Some(Self::get(self.buffers, view))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let view = self.views.nth_back(n)?;
        Some(Self::get(self.buffers, view))
    }

    /// Hoists the representation out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let buffers = self.buffers;
        self.views
            .rfold(init, |acc, view| f(acc, Self::get(buffers, view)))
    }
}

impl ExactSizeIterator for PlBinaryViewValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.views.len()
    }
}

unsafe impl TrustedLen for PlBinaryViewValuesIter<'_> {}

/// Iterator over the optional elements of a [`PlBinaryViewArray`](super::PlBinaryViewArray).
#[derive(Clone)]
pub struct PlBinaryViewIter<'a> {
    values: PlBinaryViewValuesIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlBinaryViewIter<'a> {
    /// # Safety
    /// Every view must read bytes that `buffers` holds.
    #[inline]
    pub(super) fn new(
        views: &'a [View],
        buffers: &'a [Buffer<u8>],
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: PlBinaryViewValuesIter::new(views, buffers, length),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlBinaryViewValuesIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

crate::impl_optional_iter!(PlBinaryViewIter<'a>, &'a [u8]);

#[cfg(test)]
mod tests {

    use crate::PlBinaryViewArray;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array: one inlined into its view, one that is not, and one empty.
    fn elements() -> [&'static [u8]; 3] {
        [
            b"ab",
            b"",
            b"a string longer than the twelve bytes a view inlines",
        ]
    }

    fn flat_array() -> PlBinaryViewArray {
        PlBinaryViewArray::from_iter(elements().map(Some))
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlBinaryViewArray::new_scalar(b"xy", 4);

        assert_iterates(array.values_iter(), &[b"xy".as_slice(); 4]);
        assert_iterates(array.iter(), &[Some(b"xy".as_slice()); 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlBinaryViewArray::new_scalar(b"xy", 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some(b"xy".as_slice()));
        assert_eq!(
            array.values_iter().nth_back(999_999_999),
            Some(b"xy".as_slice())
        );
        assert_eq!(array.iter().last(), Some(Some(b"xy".as_slice())));
        assert_eq!(
            array.iter().nth_back(999_999_999),
            Some(Some(b"xy".as_slice()))
        );
    }
}
