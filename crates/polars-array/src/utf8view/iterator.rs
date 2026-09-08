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

crate::impl_mapped_iter!(
    PlUtf8ViewValuesIter<'a>,
    &'a str,
    // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
    |value| unsafe { as_str(value) },
);

/// Iterator over the elements of a [`super::PlUtf8ViewArray`], `None` for the null ones.
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

crate::impl_mapped_iter!(
    PlUtf8ViewIter<'a>,
    Option<&'a str>,
    // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
    |value| value.map(|value| unsafe { as_str(value) }),
);
