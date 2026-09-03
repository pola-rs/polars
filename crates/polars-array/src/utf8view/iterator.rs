use arrow::trusted_len::TrustedLen;

use crate::binview::{PlBinaryViewIter, PlBinaryViewValuesIter};

/// The string `bytes` are.
///
/// # Safety
/// `bytes` must be valid UTF-8.
#[inline(always)]
unsafe fn as_str(bytes: &[u8]) -> &str {
    // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

/// Iterator over the elements of a [`PlUtf8ViewArray`](super::PlUtf8ViewArray), ignoring validity.
///
/// This is the iterator over the bytes under it, whose scalar views are not materialized; see
/// [`PlBinaryViewValuesIter`].
#[derive(Clone)]
pub struct PlUtf8ViewValuesIter<'a>(PlBinaryViewValuesIter<'a>);

impl<'a> PlUtf8ViewValuesIter<'a> {
    /// # Safety
    /// Every value `bytes` hands out must be valid UTF-8.
    #[inline]
    pub(super) const unsafe fn new(bytes: PlBinaryViewValuesIter<'a>) -> Self {
        Self(bytes)
    }
}

impl<'a> Iterator for PlUtf8ViewValuesIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0.next().map(|value| unsafe { as_str(value) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0.nth(n).map(|value| unsafe { as_str(value) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0.last().map(|value| unsafe { as_str(value) })
    }

    /// Folds the bytes under this iterator, which hoists their representation out of the loop.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a str) -> B,
    {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .fold(init, |acc, value| f(acc, unsafe { as_str(value) }))
    }
}

impl<'a> DoubleEndedIterator for PlUtf8ViewValuesIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0.next_back().map(|value| unsafe { as_str(value) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0.nth_back(n).map(|value| unsafe { as_str(value) })
    }

    /// Folds the bytes under this iterator, which hoists their representation out of the loop.
    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, &'a str) -> B,
    {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .rfold(init, |acc, value| f(acc, unsafe { as_str(value) }))
    }
}

impl ExactSizeIterator for PlUtf8ViewValuesIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewValuesIter<'_> {}

/// Iterator over the elements of a [`PlUtf8ViewArray`](super::PlUtf8ViewArray), `None` for the
/// null ones.
///
/// This is the iterator over the bytes under it, whose scalar views and scalar validity mask are
/// not materialized; see [`PlBinaryViewIter`].
#[derive(Clone)]
pub struct PlUtf8ViewIter<'a>(PlBinaryViewIter<'a>);

impl<'a> PlUtf8ViewIter<'a> {
    /// # Safety
    /// Every value `bytes` hands out must be valid UTF-8.
    #[inline]
    pub(super) const unsafe fn new(bytes: PlBinaryViewIter<'a>) -> Self {
        Self(bytes)
    }
}

impl<'a> Iterator for PlUtf8ViewIter<'a> {
    type Item = Option<&'a str>;

    #[inline]
    fn next(&mut self) -> Option<Option<&'a str>> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next()
            .map(|value| value.map(|value| unsafe { as_str(value) }))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Option<&'a str>> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .nth(n)
            .map(|value| value.map(|value| unsafe { as_str(value) }))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Folds the bytes under this iterator, which hoists their validity mask out of the loop.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Option<&'a str>) -> B,
    {
        self.0.fold(init, |acc, value| {
            // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
            f(acc, value.map(|value| unsafe { as_str(value) }))
        })
    }
}

impl<'a> DoubleEndedIterator for PlUtf8ViewIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Option<&'a str>> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next_back()
            .map(|value| value.map(|value| unsafe { as_str(value) }))
    }
}

impl ExactSizeIterator for PlUtf8ViewIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewIter<'_> {}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use crate::PlUtf8ViewArray;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array: one that is inlined into its view, one that is not, and one
    /// that is empty.
    fn elements() -> [&'static str; 3] {
        [
            "ab",
            "",
            "a string longer than the twelve bytes a view inlines",
        ]
    }

    fn flat_array() -> PlUtf8ViewArray {
        PlUtf8ViewArray::from_iter(elements().map(Some))
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

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(
            array.iter(),
            &[Some(elements()[0]), None, Some(elements()[2])],
        );
    }

    #[test]
    fn scalar() {
        let array = PlUtf8ViewArray::new_scalar("xy", 4);

        assert_iterates(array.values_iter(), &["xy"; 4]);
        assert_iterates(array.iter(), &[Some("xy"); 4]);
    }

    #[test]
    fn scalar_under_a_flat_mask() {
        let array = PlUtf8ViewArray::new_scalar("xy", 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert_iterates(array.iter(), &[Some("xy"), None, Some("xy")]);
    }

    #[test]
    fn all_null() {
        assert_iterates(PlUtf8ViewArray::new_full_null(3).iter(), &[None; 3]);
    }

    #[test]
    fn empty() {
        let array = PlUtf8ViewArray::new_empty();

        assert_iterates(array.values_iter(), &[]);
        assert_iterates(array.iter(), &[]);
    }

    #[test]
    fn broadcast() {
        let array = PlUtf8ViewArray::from_iter([Some(elements()[2])]);

        assert_iterates(array.broadcast_values_iter(4), &[elements()[2]; 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlUtf8ViewArray::new_scalar("xy", 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some("xy"));
        assert_eq!(array.iter().nth(999_999_999), Some(Some("xy")));
    }
}
