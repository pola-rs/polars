//! The builder of a [`PlBinaryViewArray`].

use arrow::array::View;
use arrow::bitmap::OptBitmapBuilder;
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::aliases::PlHashMap;
use polars_utils::index::ChunkId;

use super::PlBinaryViewArray;
use super::buffers::copy_value;
use crate::bitmap::PlBitmap;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};

/// A builder of a [`PlBinaryViewArray`].
#[derive(Clone)]
pub struct PlBinaryViewArrayBuilder {
    views: Vec<View>,
    /// The data buffers whose index is final: the ones already adopted or flushed, in order.
    buffers: Vec<Buffer<u8>>,
    /// The data buffers the copied bytes are written into, flushed onto `buffers` before adoption.
    active: Vec<Vec<u8>>,
    /// The index in `buffers` of every adopted buffer, keyed by the address its bytes start at.
    adopted: PlHashMap<usize, u32>,
    /// The buffer adopted last, as the address its bytes start at and the index it took.
    last_adopted: Option<(usize, u32)>,
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
            last_adopted: None,
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        builder.reserve(capacity);
        builder
    }

    /// Appends `value` as an element of its own.
    pub fn push_value(&mut self, value: &[u8]) {
        let view = self.copy_value(value);
        self.views.push(view);
        self.validity.extend_constant(1, true);
    }

    /// Appends `value` as an element of its own, leaving the validity mask untouched.
    ///
    /// The mask a caller of this builds itself — `with_validity` on the frozen array, say, out of
    /// the mask of the input it read the values off. It stays out of the loop that way: a builder
    /// this is pushed onto keeps no mask at all, where `push_value` sets a bit per element for a
    /// mask the caller then throws away.
    ///
    /// A builder must not see both this and a call that does maintain the mask
    /// ([`push_value`](Self::push_value), [`push_null`](Self::push_null), the
    /// [`StaticArrayBuilder`] methods): the bits would then stand for some of the elements and not
    /// others.
    #[inline]
    pub fn push_value_ignore_validity(&mut self, value: &[u8]) {
        let view = self.copy_value(value);
        self.views.push(view);
    }

    /// Appends a null.
    pub fn push_null(&mut self) {
        self.views.push(View::default());
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    pub fn push(&mut self, value: Option<&[u8]>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// The index the first of the buffers being written into will have.
    fn buffer_idx_offset(&self) -> u32 {
        u32::try_from(self.buffers.len())
            .expect("the built array holds more data buffers than a view can index")
    }

    /// Hands the buffers being written into over to `self.buffers`, keeping every view's index.
    fn flush_active(&mut self) {
        self.buffers.extend(self.active.drain(..).map(Buffer::from));
    }

    /// The index in `self.buffers` of `buffer`, adopting it if it is not held yet.
    fn adopt(&mut self, buffer: &Buffer<u8>) -> u32 {
        // Two views over one allocation reach it through buffers that start at the same address
        // but need not end at the same one, so what is held is the whole allocation. Expanding a
        // buffer towards the end leaves both the offsets of the views into it and the address it
        // starts at untouched, so the address keys the buffer before it is expanded — and a
        // buffer already held is answered without the refcount bump a clone would cost.
        let key = buffer.as_slice().as_ptr().addr();

        // The one buffer a run of views shares is answered out of a compare, rather than out of a
        // hash and a lookup.
        if let Some((last_key, idx)) = self.last_adopted {
            if last_key == key {
                return idx;
            }
        }

        if let Some(&idx) = self.adopted.get(&key) {
            self.last_adopted = Some((key, idx));
            return idx;
        }

        // The buffers being written into come before this one, so they take their index first.
        self.flush_active();
        let idx = self.buffer_idx_offset();
        self.buffers.push(buffer.clone().expand_end_to_storage());
        self.adopted.insert(key, idx);
        self.last_adopted = Some((key, idx));
        idx
    }

    /// A view over the bytes of `self`, holding `bytes` — copied in unless the view inlines them.
    fn copy_value(&mut self, bytes: &[u8]) -> View {
        let buffer_idx_offset = self.buffer_idx_offset();
        copy_value(&mut self.active, buffer_idx_offset, bytes)
    }

    /// The view `view` of `buffers` becomes over the data buffers of this builder.
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

    /// Appends the elements `ids` names out of `chunks`, adopting each chunk's buffers once.
    ///
    /// # Safety
    /// Every id that is not null must name a chunk of `chunks` and an element of that chunk.
    unsafe fn gather_chunks<const B: u64>(
        &mut self,
        chunks: &[&PlBinaryViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
        opt: bool,
    ) {
        // Adopting every chunk's buffers up front costs one entry per buffer whether or not any id
        // reaches it, and leaves the result holding buffers nothing in it reads. That pays only
        // when there are elements enough to amortize it: a gather of fewer elements than there are
        // chunks adopts lazily instead, one buffer per view that actually points into it.
        let hoisting_pays = ids.len() >= chunks.len();

        // Copying the bytes out of the source leaves no buffers to adopt, so there is nothing to
        // hoist and the elementwise path stands.
        if matches!(share, ShareStrategy::Never) || !hoisting_pays {
            // SAFETY: the caller's guarantee is the one these ask for.
            return unsafe {
                if opt {
                    self.opt_gather_chunks_elementwise(chunks, ids, share)
                } else {
                    self.gather_chunks_elementwise(chunks, ids, share)
                }
            };
        }

        // The buffers of every chunk, adopted once, as the indices its views are rewritten onto.
        let remaps: Vec<Vec<u32>> = chunks
            .iter()
            .map(|chunk| self.adopt_all(chunk.data_buffers()))
            .collect();

        self.views.reserve(ids.len());
        self.validity.reserve(ids.len());

        // A chunk with no mask at all has no null element, so a gather out of chunks like that
        // answers every element valid unless its own id is null — and the runs between the null
        // ids then reach the mask in one extension rather than one per element.
        let masked = chunks.iter().any(|chunk| chunk.validity().is_some());
        let mut valid_run = 0;

        for id in ids {
            if opt && id.is_null() {
                self.views.push(View::default());
                if !masked && valid_run > 0 {
                    self.validity.extend_constant(valid_run, true);
                    valid_run = 0;
                }
                self.validity.extend_constant(1, false);
                continue;
            }

            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the id is not null, so it names a chunk and an element of it, and the view
            // of an element points into that chunk's buffers, whose remapped indices are held.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                let mut view = chunk.view_unchecked(array_idx as usize);

                // An inline view holds its bytes itself, so no buffer stands behind it.
                if !view.is_inline() {
                    view.buffer_idx = *remaps
                        .get_unchecked(chunk_idx as usize)
                        .get_unchecked(view.buffer_idx as usize);
                }

                self.views.push(view);

                if masked {
                    self.validity
                        .extend_constant(1, !chunk.is_null_unchecked(array_idx as usize));
                } else {
                    valid_run += 1;
                }
            }
        }

        if !masked && valid_run > 0 {
            self.validity.extend_constant(valid_run, true);
        }
    }

    /// [`gather_chunks`](Self::gather_chunks) one element at a time.
    ///
    /// # Safety
    /// Every id must name a chunk of `chunks` and an element of that chunk.
    unsafe fn gather_chunks_elementwise<const B: u64>(
        &mut self,
        chunks: &[&PlBinaryViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        for id in ids {
            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the caller guarantees the id names an element.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                StaticArrayBuilder::extend_one(self, chunk, array_idx as usize, share);
            }
        }
    }

    /// [`gather_chunks_elementwise`](Self::gather_chunks_elementwise), a null id being a null.
    ///
    /// # Safety
    /// Every id that is not null must name a chunk of `chunks` and an element of that chunk.
    unsafe fn opt_gather_chunks_elementwise<const B: u64>(
        &mut self,
        chunks: &[&PlBinaryViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        for id in ids {
            if id.is_null() {
                StaticArrayBuilder::extend_nulls(self, 1);
                continue;
            }

            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the id is not null, so the caller guarantees it names an element.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                StaticArrayBuilder::extend_one(self, chunk, array_idx as usize, share);
            }
        }
    }

    /// The index in `self.buffers` of every buffer of `buffers`, adopting ones not held yet.
    fn adopt_all(&mut self, buffers: &Buffer<Buffer<u8>>) -> Vec<u32> {
        buffers
            .as_slice()
            .iter()
            .map(|buffer| self.adopt(buffer))
            .collect()
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
                self.validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn freeze_reset(&mut self) -> PlBinaryViewArray {
        self.flush_active();
        let views = std::mem::take(&mut self.views);
        let buffers = std::mem::take(&mut self.buffers);
        let validity = std::mem::take(&mut self.validity);
        self.adopted.clear();
        self.last_adopted = None;

        let length = views.len();
        // SAFETY: as in `freeze`.
        unsafe {
            PlBinaryViewArray::new_unchecked(
                Buffer::from(views),
                Buffer::from(buffers),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
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

    unsafe fn chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlBinaryViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        // SAFETY: forwarded with the caller's guarantee that every id names an element.
        unsafe { self.gather_chunks(chunks, ids, share, false) };
    }

    unsafe fn opt_chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlBinaryViewArray],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        // SAFETY: as above; a null id is answered with a null rather than read.
        unsafe { self.gather_chunks(chunks, ids, share, true) };
    }

    #[inline]
    unsafe fn extend_one(&mut self, other: &PlBinaryViewArray, index: usize, share: ShareStrategy) {
        // One element is taken straight over: `subslice_extend` of a single element would check
        // the subslice, resolve the values representation and go through the mask machinery per
        // element, which a gather that reads one element at a time pays for every row.
        debug_assert!(index < other.len());
        unsafe {
            self.extend_element(other, index, 1, share);
            self.validity
                .extend_constant(1, !other.is_null_unchecked(index));
        }
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
