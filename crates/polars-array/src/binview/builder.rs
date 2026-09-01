//! The builder of a [`PlBinaryViewArray`].

use arrow::array::View;
use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::aliases::PlHashMap;

use super::PlBinaryViewArray;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBinaryViewArray`].
///
/// The views are staged in a `Vec<View>`, which is what the frozen array is taken over, so the
/// array this builds is [flat](crate::Flat) — a view per element, however many of the appended
/// elements shared one. A null element is written out as a zeroed view, which reads no bytes at
/// all.
///
/// Where the bytes of an element come from is what [`ShareStrategy`] decides: appending an array
/// with [`ShareStrategy::Always`] adopts the data buffers its views point into, which costs a
/// buffer handle rather than the bytes, while [`ShareStrategy::Never`] copies the bytes into data
/// buffers of this builder's own. A buffer is adopted at most once however many arrays it is
/// appended through, and the bytes a view inlines are always what that view already stands for.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder};
/// use polars_array::{PlBinaryViewArray, PlBinaryViewArrayBuilder};
///
/// let array = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);
///
/// let mut builder = PlBinaryViewArrayBuilder::new();
/// builder.extend_nulls(1);
/// builder.extend(&array, ShareStrategy::Always);
///
/// let built = builder.freeze();
/// assert_eq!(
///     built.iter().collect::<Vec<_>>(),
///     [None, Some(b"foo".as_slice()), Some(b"bar")],
/// );
/// ```
pub struct PlBinaryViewArrayBuilder {
    views: Vec<View>,
    /// The data buffers whose index is final: the ones already adopted or flushed, in order.
    buffers: Vec<Buffer<u8>>,
    /// The data buffers the copied bytes are written into, the first of which is buffer
    /// `buffers.len()` — which is why they are flushed onto `buffers` before a buffer is adopted.
    active: Vec<Vec<u8>>,
    /// The index in `buffers` of every adopted buffer, keyed by the address its bytes start at, so
    /// that a buffer appended through more than one array is adopted once.
    adopted: PlHashMap<usize, u32>,
    validity: OptBitmapBuilder,
}

impl PlBinaryViewArrayBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            views: Vec::new(),
            buffers: Vec::new(),
            active: Vec::new(),
            adopted: PlHashMap::default(),
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        builder.reserve(capacity);
        builder
    }

    /// The index the first of the buffers being written into will have.
    fn buffer_idx_offset(&self) -> u32 {
        u32::try_from(self.buffers.len())
            .expect("the built array holds more data buffers than a view can index")
    }

    /// Hands the buffers being written into over to `self.buffers`, keeping the index every view
    /// into them already has.
    fn flush_active(&mut self) {
        self.buffers.extend(self.active.drain(..).map(Buffer::from));
    }

    /// The index in `self.buffers` of `buffer`, adopting it if it is not held yet.
    fn adopt(&mut self, buffer: &Buffer<u8>) -> u32 {
        // Two views over one allocation reach it through buffers that start at the same address
        // but need not end at the same one, so what is held is the whole allocation: expanding it
        // leaves the offsets of the views into it untouched.
        let buffer = buffer.clone().expand_end_to_storage();
        let key = buffer.as_slice().as_ptr().addr();

        if let Some(idx) = self.adopted.get(&key) {
            return *idx;
        }

        // The buffers being written into come before this one, so they take their index first.
        self.flush_active();
        let idx = self.buffer_idx_offset();
        self.buffers.push(buffer);
        self.adopted.insert(key, idx);
        idx
    }

    /// A view over the bytes of `self`, holding `bytes` — copied into the buffers being written
    /// into unless the view inlines them.
    fn copy_value(&mut self, bytes: &[u8]) -> View {
        let offset = self.buffer_idx_offset();
        View::new_with_buffers(bytes, offset, &mut self.active)
    }

    /// The view `view` of `buffers` becomes over the data buffers of this builder.
    ///
    /// A view that inlines its bytes carries them itself, so it is already what it stands for.
    /// Otherwise the buffer it points into is adopted, or the bytes it points at are copied.
    fn take_view(
        &mut self,
        mut view: View,
        buffers: &Buffer<Buffer<u8>>,
        share: ShareStrategy,
    ) -> View {
        if view.is_inline() {
            return view;
        }

        let buffer = &buffers[view.buffer_idx as usize];
        match share {
            ShareStrategy::Always => {
                view.buffer_idx = self.adopt(buffer);
                view
            },
            ShareStrategy::Never => {
                let start = view.offset as usize;
                let bytes = &buffer[start..start + view.length as usize];
                self.copy_value(bytes)
            },
        }
    }

    /// Appends the element of `other` at `i`, `repeats` times over.
    ///
    /// # Safety
    /// `i` must be smaller than `other.len()`.
    unsafe fn extend_element(
        &mut self,
        other: &PlBinaryViewArray,
        i: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        // SAFETY: `i` is in bounds of the array, so it broadcasts into the views and the mask.
        let view = unsafe { other.view_unchecked(i) };
        let is_null = unsafe { other.is_null_unchecked(i) };

        // The bytes of a null element are undetermined, so they are never copied out of it.
        let view = if is_null && matches!(share, ShareStrategy::Never) {
            View::default()
        } else {
            self.take_view(view, other.data_buffers(), share)
        };

        self.views.extend(std::iter::repeat_n(view, repeats));
    }

    /// Appends the elements of `other` at `indices`, each `repeats` times over.
    ///
    /// # Safety
    /// Every index must be smaller than `other.len()`.
    unsafe fn extend_elements(
        &mut self,
        other: &PlBinaryViewArray,
        indices: impl ExactSizeIterator<Item = usize>,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let count = indices.len() * repeats;
        self.views.reserve(count);

        // Every element of an array whose views are scalar stands for the same value, so its bytes
        // are reached — and at most copied — once, however many elements are appended. Which of
        // them are null does not come into it: the value of a null element is undetermined, so
        // writing the shared one out for it is as good as anything else.
        if let Some(view) = other.scalar_views() {
            if count > 0 {
                let view = self.take_view(view, other.data_buffers(), share);
                self.views.extend(std::iter::repeat_n(view, count));
            }
            return;
        }

        for i in indices {
            // SAFETY: the caller guarantees every index is in bounds of the array.
            unsafe { self.extend_element(other, i, repeats, share) };
        }
    }
}

impl Default for PlBinaryViewArrayBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl StaticArrayBuilder for PlBinaryViewArrayBuilder {
    type Array = PlBinaryViewArray;

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.views.len()
    }

    fn freeze(mut self) -> PlBinaryViewArray {
        self.flush_active();
        let length = self.views.len();
        // SAFETY: the views hold one slot per element, each of them rebased onto the data buffers
        // it is frozen over, and the mask was built alongside them.
        unsafe {
            PlBinaryViewArray::new_unchecked(
                Buffer::from(self.views),
                Buffer::from(self.buffers),
                length,
                self.validity.into_opt_validity(),
            )
        }
    }

    fn freeze_reset(&mut self) -> PlBinaryViewArray {
        self.flush_active();
        let views = std::mem::take(&mut self.views);
        let buffers = std::mem::take(&mut self.buffers);
        let validity = std::mem::take(&mut self.validity);
        self.adopted.clear();

        let length = views.len();
        // SAFETY: as in `freeze`.
        unsafe {
            PlBinaryViewArray::new_unchecked(
                Buffer::from(views),
                Buffer::from(buffers),
                length,
                validity.into_opt_validity(),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // A zeroed view holds no bytes at all, which the undetermined value of a null element may
        // as well be.
        self.views
            .extend(std::iter::repeat_n(View::default(), length));
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &PlBinaryViewArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        // SAFETY: the subslice was just checked against the length of the array.
        unsafe { self.extend_elements(other, start..start + length, 1, share) };
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlBinaryViewArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        // SAFETY: the subslice was just checked against the length of the array.
        unsafe { self.extend_elements(other, start..start + length, repeats, share) };
        subslice_extend_each_repeated_validity(
            &mut self.validity,
            other.validity(),
            start,
            length,
            repeats,
        );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &PlBinaryViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the caller guarantees every index is in bounds of the array.
        unsafe {
            self.extend_elements(other, idxs.iter().map(|idx| *idx as usize), 1, share);
            gather_extend_validity(&mut self.validity, other.validity(), idxs);
        }
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlBinaryViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.views.reserve(idxs.len());

        for idx in idxs {
            let idx = *idx as usize;
            if idx < other.len() {
                // SAFETY: the index was just checked against the length of the array.
                unsafe { self.extend_element(other, idx, 1, share) };
            } else {
                self.views.push(View::default());
            }
        }

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value of more than `View::MAX_INLINE_SIZE` bytes, which no view inlines.
    const LONG: &[u8] = b"a value that is too long to inline";

    #[test]
    fn appending_subslices_and_repeats() {
        let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None, Some(LONG)]
            .into_iter()
            .collect();

        let mut builder = PlBinaryViewArrayBuilder::with_capacity(8);
        builder.subslice_extend(&array, 1, 2, ShareStrategy::Always);
        builder.subslice_extend_repeated(&array, 0, 2, 2, ShareStrategy::Never);
        builder.subslice_extend_each_repeated(&array, 2, 1, 2, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [
                None,
                Some(LONG),
                Some(b"foo".as_slice()),
                None,
                Some(b"foo".as_slice()),
                None,
                Some(LONG),
                Some(LONG),
            ],
        );
    }

    #[test]
    fn gathering() {
        let array: PlBinaryViewArray = [Some(b"foo".as_slice()), None, Some(LONG)]
            .into_iter()
            .collect();

        let mut builder = PlBinaryViewArrayBuilder::new();
        unsafe { builder.gather_extend(&array, &[2, 0, 1], ShareStrategy::Always) };
        builder.opt_gather_extend(&array, &[2, 9], ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(
            built.iter().collect::<Vec<_>>(),
            [Some(LONG), Some(b"foo".as_slice()), None, Some(LONG), None],
        );
    }

    #[test]
    fn shared_buffers_are_adopted_once() {
        let array = PlBinaryViewArray::from_values_iter([LONG, b"foo".as_slice()]);
        assert_eq!(array.data_buffers().len(), 1);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.extend(&array, ShareStrategy::Always);
        builder.extend(&array, ShareStrategy::Always);
        builder.extend(&array.clone().sliced(0, 1), ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 5);
        assert_eq!(
            built.data_buffers().len(),
            1,
            "one allocation is adopted once, however many arrays it is appended through",
        );
        assert!(
            built.data_buffers()[0].is_same_buffer(&array.data_buffers()[0]),
            "the bytes must be shared, not copied",
        );
        assert_eq!(built.value(4), LONG);
    }

    #[test]
    fn copied_buffers_are_the_builders_own() {
        let array = PlBinaryViewArray::from_values_iter([LONG]);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.extend(&array, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(built.value(0), LONG);
        assert!(
            !built.data_buffers()[0].is_same_buffer(&array.data_buffers()[0]),
            "the bytes must be copied, not shared",
        );
    }

    #[test]
    fn copied_and_adopted_buffers_are_indexed_together() {
        let copied = PlBinaryViewArray::from_values_iter([LONG]);
        let adopted = PlBinaryViewArray::from_values_iter([b"another value too long to inline"]);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.extend(&copied, ShareStrategy::Never);
        builder.extend(&adopted, ShareStrategy::Always);
        builder.extend(&copied, ShareStrategy::Never);

        let built = builder.freeze();
        assert_eq!(built.value(0), LONG);
        assert_eq!(built.value(1), b"another value too long to inline");
        assert_eq!(built.value(2), LONG);
    }

    #[test]
    fn a_scalar_array_is_appended_without_being_materialized() {
        let array = PlBinaryViewArray::new_scalar(LONG, 1_000_000_000);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.subslice_extend(&array, 999_999_997, 3, ShareStrategy::Never);
        unsafe { builder.gather_extend(&array, &[0, 999_999_999], ShareStrategy::Always) };

        let built = builder.freeze();
        assert_eq!(built.len(), 5);
        assert!(built.iter().all(|value| value == Some(LONG)));
        assert_eq!(
            built.data_buffers().len(),
            2,
            "the bytes of the one value are copied once and shared once",
        );
    }

    #[test]
    fn a_fully_null_array_appends_its_nulls() {
        let array = PlBinaryViewArray::new_full_null(1_000_000_000);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.subslice_extend(&array, 0, 3, ShareStrategy::Always);

        let built = builder.freeze();
        assert_eq!(built.len(), 3);
        assert_eq!(built.null_count(), 3);
        assert!(built.data_buffers().is_empty());
    }

    #[test]
    fn freeze_reset_leaves_an_empty_builder() {
        let array = PlBinaryViewArray::from_values_iter([LONG]);

        let mut builder = PlBinaryViewArrayBuilder::new();
        builder.extend(&array, ShareStrategy::Never);
        let built = builder.freeze_reset();
        assert_eq!(built.value(0), LONG);

        assert!(builder.is_empty());
        builder.extend(&array, ShareStrategy::Never);
        let built = builder.freeze();
        assert_eq!(built.len(), 1);
        assert_eq!(built.value(0), LONG);
    }
}
