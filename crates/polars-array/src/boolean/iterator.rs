use arrow::trusted_len::TrustedLen;

use crate::bitmap::{PlBitmapIter, PlBitmapRef, ValidityFold, ValidityIter};

/// Iterator over the optional elements of a [`PlBooleanArray`](super::PlBooleanArray).
#[derive(Clone)]
pub struct PlBooleanIter<'a> {
    values: PlBitmapIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlBooleanIter<'a> {
    /// # Panics
    /// Panics unless `values` and `validity` both have `length` bits.
    #[inline]
    pub(super) fn new(
        values: PlBitmapRef<'a>,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert_eq!(values.len(), length);
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: values.iter(),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlBitmapIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

impl Iterator for PlBooleanIter<'_> {
    type Item = Option<bool>;

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

    /// Hoists the validity mask out of the loop, and the representation of both masks with it.
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

impl DoubleEndedIterator for PlBooleanIter<'_> {
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

impl ExactSizeIterator for PlBooleanIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }
}

unsafe impl TrustedLen for PlBooleanIter<'_> {}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use crate::PlBooleanArray;
    use crate::bitmap::PlBitmap;
    use crate::iterator_tests::assert_iterates;

    #[test]
    fn flat() {
        let array = PlBooleanArray::from_vec(vec![true, false, true]);

        assert_iterates(array.values_iter(), &[true, false, true]);
        assert_iterates(array.iter(), &[Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn scalar() {
        let array = PlBooleanArray::new_scalar(true, 4);

        assert_iterates(array.values_iter(), &[true; 4]);
        assert_iterates(array.iter(), &[Some(true); 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlBooleanArray::new_scalar(true, 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.iter().nth(999_999_999), Some(Some(true)));
        assert_eq!(array.iter().nth_back(999_999_999), Some(Some(true)));
        assert_eq!(array.iter().len(), 1_000_000_000);
    }

    /// A mask of mixed bits, which is read by position alongside the values rather than walked
    /// alongside them.
    #[test]
    fn mixed_validity() {
        let array = PlBooleanArray::from_vec(vec![true, false, true, false]).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true, true])),
        ));

        assert_iterates(array.values_iter(), &[true, false, true, false]);
        assert_iterates(array.iter(), &[Some(true), None, Some(true), Some(false)]);
    }

    /// Both bitmaps of a sliced array start partway into the bytes backing them.
    #[test]
    fn a_sliced_mask_is_read_from_its_own_front() {
        let array = PlBooleanArray::from_vec(vec![true, true, false, true, false])
            .with_validity(Some(PlBitmap::from_bitmap(Bitmap::from_iter([
                true, true, false, true, false,
            ]))))
            .sliced(2, 3);

        assert_iterates(array.values_iter(), &[false, true, false]);
        assert_iterates(array.iter(), &[None, Some(true), None]);
    }
}
