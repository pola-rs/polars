use crate::prelude::*;
use crate::series::IsSorted;

/// The order `ca` stands in once it is reversed: an ascending column read backwards is a
/// descending one, and the other way about.
fn reversed_sorted_flag<T: PolarsDataType>(ca: &ChunkedArray<T>) -> IsSorted {
    match ca.is_sorted_flag() {
        IsSorted::Ascending => IsSorted::Descending,
        IsSorted::Descending => IsSorted::Ascending,
        IsSorted::Not => IsSorted::Not,
    }
}

/// A chunked array that is its own reverse, if it is one: a column repeating one element.
///
/// The elements come back in the places they were already in, but the sorted flag still turns
/// over. One element repeated stands in both orders at once, so either flag is the truth — and
/// `sort_with`'s fast path reaches `reverse` precisely to be handed the *other* one, so a caller
/// that asked for a descending sort of an ascending column must not get an ascending one back.
fn reverses_to_itself<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<ChunkedArray<T>> {
    let repeats = match ca.chunks().as_slice() {
        [chunk] => PlArray::is_scalar(&**chunk),
        // Several chunks that all repeat the same element hold it in every place too, so the
        // reversed column is the one that came in — with the chunks in the other order, which
        // says nothing about the elements.
        _ => ca.repeats_one_element(),
    };

    repeats.then(|| ca.with_sorted_flag(reversed_sorted_flag(ca)))
}

/// Reverses `ca` a chunk at a time: every chunk reversed by `reversed`, in the opposite order.
///
/// Each chunk keeps whichever representation it is in, so a column of repeated chunks stays
/// repeated and nothing is written out for it.
fn reverse_chunk_wise<T, F>(ca: &ChunkedArray<T>, reversed: F) -> ChunkedArray<T>
where
    T: PolarsDataType,
    F: Fn(&T::Array) -> Box<dyn PlArray>,
{
    let chunks = ca.downcast_iter().rev().map(reversed).collect::<Vec<_>>();
    debug_assert!(
        !chunks.is_empty(),
        "a chunked array holds at least one chunk"
    );

    // SAFETY: reversing keeps every chunk's dtype and the column's length.
    unsafe {
        ChunkedArray::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, ca.dtype().clone())
    }
}

impl<T> ChunkReverse for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn reverse(&self) -> ChunkedArray<T> {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

        // Reversing each chunk and then the order they come in reverses the column, without ever
        // reading an element out of it: each chunk's values buffer is reversed in one pass, and a
        // chunk that repeats one element is handed back as it is. Collecting through the column's
        // own iterator instead cost 8x on eight chunks, which no single-chunk fast path reaches.
        let mut out = reverse_chunk_wise(self, |arr| arr.reversed().into_boxed());
        out.rename(self.name().clone());

        out.set_sorted_flag(reversed_sorted_flag(self));

        out
    }
}

macro_rules! impl_reverse {
    ($arrow_type:ident, $ca_type:ident) => {
        impl ChunkReverse for $ca_type {
            fn reverse(&self) -> Self {
                if self.is_empty() {
                    return self.clone();
                };
                if let Some(ca) = reverses_to_itself(self) {
                    return ca;
                }
                let mut ca: Self = self.iter().rev().collect_trusted();
                ca.rename(self.name().clone());
                ca
            }
        }
    };
}

impl_reverse!(BinaryOffsetType, BinaryOffsetChunked);

impl ChunkReverse for BooleanChunked {
    fn reverse(&self) -> Self {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

        // Both of a boolean chunk's axes are bitmaps, and a bitmap reverses a word at a time, so
        // reversing chunk-wise reads no element out of the column at all.
        let mut ca = reverse_chunk_wise(self, |arr| arr.reversed().into_boxed());
        ca.rename(self.name().clone());
        ca
    }
}

impl ChunkReverse for ListChunked {
    fn reverse(&self) -> Self {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }
        if self.is_empty() {
            return self.clone();
        };

        // Read out of the chunks by index, rather than collected back from a `Series` per element:
        // a collect carries no inner type of its own, so a column of nothing but nulls came back
        // as a `List(Null)` — the elements alone do not say what is under them.
        let idx = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            (0..self.len() as IdxSize).rev().collect(),
        );
        // SAFETY: every index is below the length.
        let mut ca = unsafe { self.take_unchecked(&idx) };
        ca.rename(self.name().clone());
        ca
    }
}

impl ChunkReverse for BinaryChunked {
    fn reverse(&self) -> Self {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

        // The views are reordered one per element, but they index a side table that the order of
        // the elements says nothing about, so the buffers holding the bytes are carried over
        // untouched. Gathering by reversed indices instead rebuilt those buffers: 21x on eight
        // chunks.
        let mut ca = reverse_chunk_wise(self, |arr| arr.reversed().into_boxed());
        ca.rename(self.name().clone());
        ca
    }
}

impl ChunkReverse for StringChunked {
    fn reverse(&self) -> Self {
        unsafe { self.as_binary().reverse().to_string_unchecked() }
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkReverse for ArrayChunked {
    fn reverse(&self) -> Self {
        if let Some(ca) = reverses_to_itself(self) {
            return ca;
        }

        // Read out of the chunks by index, as `ListChunked` does: the builder this used to push
        // into only exists for a numeric inner type, so every other one — a string, a boolean, a
        // list, a struct — reached a `todo!()` and panicked.
        let idx = IdxCa::from_vec(
            PlSmallStr::EMPTY,
            (0..self.len() as IdxSize).rev().collect(),
        );
        // SAFETY: every index is below the length.
        let mut ca = unsafe { self.take_unchecked(&idx) };
        ca.rename(self.name().clone());
        ca
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkReverse for ObjectChunked<T> {
    fn reverse(&self) -> Self {
        // SAFETY: we know we don't go out of bounds.
        unsafe {
            self.take_unchecked(
                &(0..self.len() as IdxSize)
                    .rev()
                    .collect_ca(PlSmallStr::EMPTY),
            )
        }
    }
}
