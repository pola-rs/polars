//! The builder of a [`PlUtf8ViewArray`].

use polars_utils::IdxSize;

use super::PlUtf8ViewArray;
use crate::binview::{PlBinaryViewArray, PlBinaryViewArrayBuilder};
use crate::builder::{ShareStrategy, StaticArrayBuilder};

/// A builder of a [`PlUtf8ViewArray`].
#[derive(Default)]
pub struct PlUtf8ViewArrayBuilder(PlBinaryViewArrayBuilder);

impl PlUtf8ViewArrayBuilder {
    /// A builder with no capacity reserved.
    #[inline]
    pub fn new() -> Self {
        Self(PlBinaryViewArrayBuilder::new())
    }

    /// A builder with room for `capacity` elements.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(PlBinaryViewArrayBuilder::with_capacity(capacity))
    }

    /// The builder of the bytes under these strings.
    ///
    /// # Safety
    /// Every value appended through it must be valid UTF-8.
    #[inline]
    pub unsafe fn inner_mut(&mut self) -> &mut PlBinaryViewArrayBuilder {
        &mut self.0
    }

    /// Appends a value.
    #[inline]
    pub fn push_value(&mut self, value: &str) {
        self.0.push_value(value.as_bytes());
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        self.0.push_null();
    }

    /// Appends a value, or a null if there is none.
    #[inline]
    pub fn push(&mut self, value: Option<&str>) {
        self.0.push(value.map(str::as_bytes));
    }
}

/// Borrows an array of strings as the bytes under them.
///
/// This is [`PlUtf8ViewArray::as_binview`] as a free function, so that it can be passed to the
/// inner builder without shadowing anything.
#[inline(always)]
fn as_binview(array: &PlUtf8ViewArray) -> &PlBinaryViewArray {
    array.as_binview()
}

impl StaticArrayBuilder for PlUtf8ViewArrayBuilder {
    type Array = PlUtf8ViewArray;

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn freeze(self) -> PlUtf8ViewArray {
        // SAFETY: every value appended was a `&str`.
        unsafe { PlUtf8ViewArray::from_binview_unchecked(self.0.freeze()) }
    }

    #[inline]
    fn freeze_reset(&mut self) -> PlUtf8ViewArray {
        // SAFETY: every value appended was a `&str`.
        unsafe { PlUtf8ViewArray::from_binview_unchecked(self.0.freeze_reset()) }
    }

    #[inline]
    fn extend_nulls(&mut self, length: usize) {
        self.0.extend_nulls(length);
    }

    #[inline]
    fn subslice_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.0
            .subslice_extend(as_binview(other), start, length, share);
    }

    #[inline]
    fn subslice_extend_repeated(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.0
            .subslice_extend_repeated(as_binview(other), start, length, repeats, share);
    }

    #[inline]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.0
            .subslice_extend_each_repeated(as_binview(other), start, length, repeats, share);
    }

    #[inline]
    unsafe fn gather_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the caller keeps every index in bounds.
        unsafe { self.0.gather_extend(as_binview(other), idxs, share) };
    }

    #[inline]
    fn opt_gather_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.0.opt_gather_extend(as_binview(other), idxs, share);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value of more than `View::MAX_INLINE_SIZE` bytes, which no view inlines.
    const LONG: &str = "a value that is too long to inline";

    #[test]
    fn appending_values_and_arrays() {
        let array: PlUtf8ViewArray = [Some("foo"), None, Some(LONG)].into_iter().collect();

        let mut builder = PlUtf8ViewArrayBuilder::with_capacity(8);
        builder.push_value("bar");
        builder.push_null();
        builder.push(Some("baz"));
        builder.extend_nulls(1);
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Always);
        builder.subslice_extend_repeated(&array, 0, 1, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Never);

        assert_eq!(builder.len(), 10);
        assert_eq!(
            builder.freeze().iter().collect::<Vec<_>>(),
            [
                Some("bar"),
                None,
                Some("baz"),
                None,
                None,
                Some(LONG),
                Some("foo"),
                Some("foo"),
                Some(LONG),
                Some(LONG),
            ],
        );
    }

    #[test]
    fn gathering() {
        let array: PlUtf8ViewArray = [Some("foo"), None, Some(LONG)].into_iter().collect();

        let mut builder = PlUtf8ViewArrayBuilder::new();
        builder.reserve(4);
        // SAFETY: every index is in bounds of the array.
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[2, 9], ShareStrategy::Never);

        assert_eq!(
            builder.freeze_reset().iter().collect::<Vec<_>>(),
            [Some(LONG), Some("foo"), None, Some(LONG), None],
        );
        // Freezing resets the builder, leaving it empty rather than dropping it.
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn sharing_the_bytes_of_an_appended_array() {
        let array = PlUtf8ViewArray::new_scalar(LONG, 2);
        assert_eq!(array.data_buffers().len(), 1);

        let mut builder = PlUtf8ViewArrayBuilder::new();
        builder.extend(&array, ShareStrategy::Always);

        let built = builder.freeze();
        assert!(
            built.data_buffers()[0].is_same_buffer(&array.data_buffers()[0]),
            "the bytes must be shared, not copied",
        );

        // Copying them out instead leaves the built array holding its own.
        let mut builder = PlUtf8ViewArrayBuilder::new();
        builder.extend(&array, ShareStrategy::Never);

        let built = builder.freeze();
        assert!(!built.data_buffers()[0].is_same_buffer(&array.data_buffers()[0]));
        assert_eq!(built.value(1), LONG);
    }
}
