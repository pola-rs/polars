use crate::prelude::*;
use crate::series::IsSorted;

/// The order `ca` stands in once it is reversed.
fn reversed_sorted_flag<T: PolarsDataType>(ca: &ChunkedArray<T>) -> IsSorted {
    match ca.is_sorted_flag() {
        IsSorted::Ascending => IsSorted::Descending,
        IsSorted::Descending => IsSorted::Ascending,
        IsSorted::Not => IsSorted::Not,
    }
}

/// A chunked array that is its own reverse, if it is one: a column repeating one element.
fn reverses_to_itself<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<ChunkedArray<T>> {
    let repeats = match ca.chunks().as_slice() {
        [chunk] => PlArray::is_scalar(&**chunk),
        _ => ca.repeats_one_element(),
    };

    repeats.then(|| ca.with_sorted_flag(reversed_sorted_flag(ca)))
}

/// Reverses `ca` a chunk at a time: every chunk reversed by `reversed`, in the opposite order.
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
