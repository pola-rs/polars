//! What the `distinct` family answers of a chunked array whose one chunk repeats a single element.

use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;

/// `length`, if `chunks` is a single chunk that repeats one element over more than one of them.
///
/// Every element of such a chunk is the same one — the same value throughout, or a null
/// throughout — which settles the whole `distinct` family on it without a single element being
/// hashed: the first element is the only one that is distinct in it, so is the last, and none of
/// them occurs just once.
fn repeated_element(chunks: &[PlArrayRef], length: usize) -> Option<usize> {
    let [chunk] = chunks else {
        return None;
    };

    (length > 1 && chunk.is_scalar()).then_some(length)
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
