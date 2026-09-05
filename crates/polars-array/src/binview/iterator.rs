use arrow::array::View;
use arrow::trusted_len::TrustedLen;
use polars_buffer::Buffer;
use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

use crate::bitmap::{PlBitmapRef, ValidityFold, ValidityIter};
use crate::broadcast::broadcast_slice;

/// Iterator over the values of a [`PlBinaryViewArray`](super::PlBinaryViewArray), ignoring
/// validity.
#[derive(Clone)]
pub struct PlBinaryViewValuesIter<'a> {
    views: SliceBroadcastIter<'a, View>,
    buffers: &'a [Buffer<u8>],
}

impl<'a> PlBinaryViewValuesIter<'a> {
    /// # Panics
    /// Panics unless `views` is flat or scalar for `length`, per [`crate::broadcast`].
    ///
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

    /// Hoists the representation out of the loop: flat views fold as the slice they are, and scalar
    /// ones fold over the single view they hold.
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
    /// # Panics
    /// Panics unless `views` is flat or scalar for `length`, per [`crate::broadcast`], and
    /// `validity` has `length` bits.
    ///
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

impl<'a> Iterator for PlBinaryViewIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.values.next()?;
        Some(self.validity.next().then_some(value))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the values, whether or not there is a value left.
        let is_valid = self.validity.nth(n);
        let value = self.values.nth(n)?;
        Some(is_valid.then_some(value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.values.count()
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the validity mask out of the loop, and the representation of the views with it.
    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, mask) = self.split();
        // SAFETY: the mask has one bit per element, and the values and the mask are walked in
        // lockstep, so it has a bit for every value left to yield.
        unsafe { mask.fold_values(values, init, f) }
    }
}

impl DoubleEndedIterator for PlBinaryViewIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let value = self.values.next_back()?;
        Some(self.validity.next_back().then_some(value))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the values, whether or not there is a value left.
        let is_valid = self.validity.nth_back(n);
        let value = self.values.nth_back(n)?;
        Some(is_valid.then_some(value))
    }

    /// Hoists the validity mask out of the loop, the way [`Iterator::fold`] does.
    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (values, mask) = self.split();
        // SAFETY: the mask has a bit for every value left to yield, per `Iterator::fold`.
        unsafe { mask.rfold_values(values, init, f) }
    }
}

impl ExactSizeIterator for PlBinaryViewIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }
}

unsafe impl TrustedLen for PlBinaryViewIter<'_> {}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use crate::PlBinaryViewArray;
    use crate::bitmap::PlBitmap;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array: one that is inlined into its view, one that is not, and one
    /// that is empty.
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

    /// A mask of mixed bits, which is read by position alongside the views.
    #[test]
    fn mixed_validity() {
        let array = flat_array().with_validity(Some(PlBitmap::from_bitmap(Bitmap::from_iter([
            true, false, true,
        ]))));

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(
            array.iter(),
            &[Some(elements()[0]), None, Some(elements()[2])],
        );
    }
}
