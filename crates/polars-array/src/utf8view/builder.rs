//! The builder of a [`PlUtf8ViewArray`].

use polars_utils::IdxSize;
use polars_utils::index::ChunkId;

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
    unsafe fn extend_one(&mut self, other: &PlUtf8ViewArray, index: usize, share: ShareStrategy) {
        // Forwarded explicitly: the default would go back through `subslice_extend`, which is
        // what the override on the inner builder exists to skip.
        unsafe { self.0.extend_one(other.as_binview(), index, share) };
    }

    unsafe fn chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlUtf8ViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        // Forwarded explicitly, as `extend_one` is: the default would not reach the inner
        // builder's override, which is where the buffer bookkeeping is hoisted out of the loop.
        let chunks: Vec<&PlBinaryViewArray> = chunks.iter().map(|c| c.as_binview()).collect();
        // SAFETY: forwarded with the caller's guarantee that every id names an element.
        unsafe { self.0.chunked_gather_extend(&chunks, ids, share) };
    }

    unsafe fn opt_chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlUtf8ViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        // As above.
        let chunks: Vec<&PlBinaryViewArray> = chunks.iter().map(|c| c.as_binview()).collect();
        // SAFETY: as above; a null id is answered with a null rather than read.
        unsafe { self.0.opt_chunked_gather_extend(&chunks, ids, share) };
    }

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
