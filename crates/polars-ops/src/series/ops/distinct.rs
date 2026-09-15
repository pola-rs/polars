//! What the `distinct` family answers of a chunked array whose one chunk repeats a single element.

use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;

/// `length`, if every one of the `length` elements `chunks` holds is the same one.
///
/// A single chunk that repeats one element is the same value throughout, or the same null
/// throughout; so is a column with nothing but nulls in it, whatever its chunks look like.
/// Either way the whole `distinct` family is settled without a single element being hashed:
/// the first element is the only one that is distinct, so is the last, and none of them occurs
/// just once.
fn repeated_element(chunks: &[PlArrayRef], length: usize) -> Option<usize> {
    if length <= 1 {
        return None;
    }

    // Nulls are all the same element, so a column of nothing else holds one element throughout
    // however many chunks it is spread over; the counts are the ones the masks already carry.
    if chunks.iter().map(|chunk| chunk.null_count()).sum::<usize>() == length {
        return Some(length);
    }

    let [chunk] = chunks else {
        return None;
    };

    chunk.is_scalar().then_some(length)
}

/// [`repeated_element`], for a chunked array.
pub(super) fn repeated_element_len<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<usize> {
    repeated_element(ca.chunks(), ca.len())
}

/// [`repeated_element`], for a series whose type is only known at runtime — a nested one, whose
/// `distinct` answers are otherwise read off the groups its rows fall into.
pub(super) fn repeated_element_len_series(s: &Series) -> Option<usize> {
    repeated_element(s.chunks(), s.len())
}

/// A mask of `length` elements that is set at `index` alone.
pub(super) fn only(name: PlSmallStr, length: usize, index: usize) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(length);
    out.extend_constant(length, false);
    out.set(index, true);

    BooleanChunked::from_bitmap(name, out.into())
}
