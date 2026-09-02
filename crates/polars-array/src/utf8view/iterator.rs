use arrow::trusted_len::TrustedLen;

use crate::binview::{PlBinaryViewIter, PlBinaryViewValuesIter};

/// Iterator over the elements of a [`PlUtf8ViewArray`](super::PlUtf8ViewArray), ignoring validity.
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
        self.0
            .next()
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for PlUtf8ViewValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next_back()
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }
}

impl ExactSizeIterator for PlUtf8ViewValuesIter<'_> {}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewValuesIter<'_> {}

/// Iterator over the elements of a [`PlUtf8ViewArray`](super::PlUtf8ViewArray), `None` for the
/// null ones.
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
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for PlUtf8ViewIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next_back()
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }
}

impl ExactSizeIterator for PlUtf8ViewIter<'_> {}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewIter<'_> {}
