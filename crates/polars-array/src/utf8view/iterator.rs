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

#[cfg(test)]
mod tests {

    use crate::PlUtf8ViewArray;
    use crate::iterator_tests::assert_iterates;

    /// The elements of a flat array: one inlined into its view, one that is not, and one empty.
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
    fn scalar() {
        let array = PlUtf8ViewArray::new_scalar("xy", 4);

        assert_iterates(array.values_iter(), &["xy"; 4]);
        assert_iterates(array.iter(), &[Some("xy"); 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlUtf8ViewArray::new_scalar("xy", 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.values_iter().nth(999_999_999), Some("xy"));
        assert_eq!(array.values_iter().nth_back(999_999_999), Some("xy"));
        assert_eq!(array.iter().nth(999_999_999), Some(Some("xy")));
        assert_eq!(array.iter().nth_back(999_999_999), Some(Some("xy")));
    }
}
