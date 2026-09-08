//! The builder of a [`PlPrimitiveArray`].

use arrow::bitmap::OptBitmapBuilder;
use arrow::types::{AlignedBytes, NativeType};
use polars_utils::IdxSize;
use polars_utils::index::ChunkId;
use polars_utils::vec::PushUnchecked;

use super::PlPrimitiveArray;
use super::bytes::{self, Bytes};
use crate::bitmap::PlBitmap;
use crate::builder::{
    ShareStrategy, StaticArrayBuilder, assert_subslice, gather_extend_validity,
    opt_gather_extend_validity, subslice_extend_each_repeated_validity, subslice_extend_validity,
};
use crate::static_array::StaticArray;

/// A builder of a [`PlPrimitiveArray`].
#[derive(Clone)]
pub struct PlPrimitiveArrayBuilder<T: NativeType> {
    values: Vec<Bytes<T>>,
    validity: OptBitmapBuilder,
}

impl<T: NativeType> PlPrimitiveArrayBuilder<T> {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            validity: OptBitmapBuilder::default(),
        }
    }

    /// Creates an empty builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self::new();
        builder.reserve(capacity);
        builder
    }

    /// A builder holding `values` and `validity` as the elements appended so far.
    pub(crate) fn from_parts(values: Vec<Bytes<T>>, validity: OptBitmapBuilder) -> Self {
        Self { values, validity }
    }

    /// Appends `value` as an element of its own.
    #[inline]
    pub fn push_value(&mut self, value: T) {
        self.values.push(bytes::to_bytes(value));
        self.validity.extend_constant(1, true);
    }

    /// Appends every value `values` yields, in order, none of them null.
    #[inline]
    pub fn push_values<I: IntoIterator<Item = T>>(&mut self, values: I) {
        let before = self.values.len();
        self.values
            .extend(values.into_iter().map(bytes::to_bytes::<T>));
        self.validity
            .extend_constant(self.values.len() - before, true);
    }

    /// Appends a null.
    #[inline]
    pub fn push_null(&mut self) {
        // The value of a null element is undetermined, so anything at all does.
        self.values.push(Bytes::<T>::zeros());
        self.validity.extend_constant(1, false);
    }

    /// Appends `value`, or a null if it is [`None`].
    #[inline]
    pub fn push(&mut self, value: Option<T>) {
        match value {
            Some(value) => self.push_value(value),
            None => self.push_null(),
        }
    }

    /// Appends the values the ids name, reading nothing but the values of the chunks.
    ///
    /// # Safety
    /// Room for `ids.len()` more values must be reserved, and every id must name an element.
    unsafe fn gather_values<const B: u64>(
        &mut self,
        chunks: &[&PlPrimitiveArray<T>],
        ids: &[ChunkId<B>],
    ) {
        // Reading the values out of slices avoids the buffer indirection — and, for chunks that
        // may be scalar, the per-element `broadcast_index` — that `value_unchecked` pays. A chunk
        // that holds one slot per element has a slice; one that repeats a single value does not,
        // and then every chunk is read through the array instead.
        let Some(slices) = chunks
            .iter()
            .map(|chunk| chunk.as_slice())
            .collect::<Option<Vec<&[T]>>>()
        else {
            for id in ids {
                let (chunk_idx, array_idx) = id.extract();
                // SAFETY: the caller guarantees the id names an element, and room for it.
                unsafe {
                    let chunk = chunks.get_unchecked(chunk_idx as usize);
                    let value = chunk.value_unchecked(array_idx as usize);
                    self.values.push_unchecked(bytes::to_bytes(value));
                }
            }
            return;
        };

        for id in ids {
            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: as above; the slices hold one slot per element of their chunk.
            unsafe {
                let slice = slices.get_unchecked(chunk_idx as usize);
                let value = *slice.get_unchecked(array_idx as usize);
                self.values.push_unchecked(bytes::to_bytes(value));
            }
        }
    }

    /// Appends the `length` values of `other` starting at `start`, ignoring its validity mask.
    fn extend_values(&mut self, other: &PlPrimitiveArray<T>, start: usize, length: usize) {
        bytes::extend_subslice(&mut self.values, other.values_bytes(), start, length);
    }
}

impl<T: NativeType> Default for PlPrimitiveArrayBuilder<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativeType> StaticArrayBuilder for PlPrimitiveArrayBuilder<T> {
    type Array = PlPrimitiveArray<T>;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    fn freeze(self) -> PlPrimitiveArray<T> {
        let length = self.values.len();
        // SAFETY: the values hold one slot per element, and so does the mask that was built
        // alongside them.
        unsafe {
            PlPrimitiveArray::new_unchecked(
                bytes::buffer_from_byte_vec::<T>(self.values),
                length,
                self.validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn freeze_reset(&mut self) -> PlPrimitiveArray<T> {
        let values = std::mem::take(&mut self.values);
        let validity = std::mem::take(&mut self.validity);
        let length = values.len();
        // SAFETY: as in `freeze`.
        unsafe {
            PlPrimitiveArray::new_unchecked(
                bytes::buffer_from_byte_vec::<T>(values),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }

    fn extend_nulls(&mut self, length: usize) {
        // The value of a null element is undetermined, so anything at all does.
        bytes::extend_undetermined(&mut self.values, length);
        self.validity.extend_constant(length, false);
    }

    #[inline]
    unsafe fn extend_one(
        &mut self,
        other: &PlPrimitiveArray<T>,
        index: usize,
        _share: ShareStrategy,
    ) {
        // A single value is pushed straight onto the buffers: going through `subslice_extend`
        // would cost a call into the out-of-line byte-class core per element.
        debug_assert!(index < other.len());
        self.push(unsafe { other.get_unchecked(index) });
    }

    unsafe fn chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlPrimitiveArray<T>],
        ids: &[ChunkId<B>],
        _share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        // A chunk with no mask at all has no null element, so a gather out of chunks like that
        // answers every element valid — one extension of the mask rather than one per element,
        // and the mask is never read while the values are gathered.
        if chunks.iter().any(|chunk| chunk.validity().is_some()) {
            for id in ids {
                let (chunk_idx, array_idx) = id.extract();
                // SAFETY: the caller guarantees the id names an element.
                unsafe {
                    let chunk = chunks.get_unchecked(chunk_idx as usize);
                    self.push(chunk.get_unchecked(array_idx as usize));
                }
            }
            return;
        }

        // SAFETY: room for every element was reserved above, and the caller guarantees every id
        // names an element of the chunk it points at.
        unsafe { self.gather_values(chunks, ids) };
        self.validity.extend_constant(ids.len(), true);
    }

    unsafe fn opt_chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&PlPrimitiveArray<T>],
        ids: &[ChunkId<B>],
        _share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        let masked = chunks.iter().any(|chunk| chunk.validity().is_some());

        for id in ids {
            if id.is_null() {
                self.push_null();
                continue;
            }

            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the id is not null, so the caller guarantees it names an element.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                let index = array_idx as usize;

                // An unmasked chunk answers every element of its own valid, which leaves the id
                // as the only thing that can make one null.
                if masked {
                    self.push(chunk.get_unchecked(index));
                } else {
                    self.push_value(chunk.value_unchecked(index));
                }
            }
        }
    }

    fn subslice_extend(
        &mut self,
        other: &PlPrimitiveArray<T>,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        self.extend_values(other, start, length);
        subslice_extend_validity(&mut self.validity, other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlPrimitiveArray<T>,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);

        bytes::extend_subslice_each_repeated(
            &mut self.values,
            other.values_bytes(),
            start,
            length,
            repeats,
        );

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
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of the array, and therefore of its values.
        unsafe { bytes::extend_gathered(&mut self.values, other.values_bytes(), idxs) };

        // SAFETY: the indices are in bounds of the array, and therefore of its mask.
        unsafe { gather_extend_validity(&mut self.validity, other.validity(), idxs) };
    }

    fn opt_gather_extend(
        &mut self,
        other: &PlPrimitiveArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        bytes::extend_opt_gathered(&mut self.values, other.values_bytes(), other.len(), idxs);

        opt_gather_extend_validity(&mut self.validity, other.validity(), idxs, other.len());
    }
}
