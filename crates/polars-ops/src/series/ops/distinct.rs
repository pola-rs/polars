//! What the `distinct` family answers of a chunked array whose one chunk repeats a single element.

use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;

/// `length`, if every one of the `length` elements `chunks` holds is the same one.
///
/// A column with nothing but nulls in it holds the same null throughout, whatever its chunks look
/// like. Either way the whole `distinct` family is settled without a single element being hashed:
/// the first element is the only one that is distinct, so is the last, and none of them occurs
/// just once.
fn all_null(chunks: &[PlArrayRef], length: usize) -> Option<usize> {
    if length <= 1 {
        return None;
    }

    // Nulls are all the same element, so a column of nothing else holds one element throughout
    // however many chunks it is spread over; the counts are the ones the masks already carry.
    (chunks.iter().map(|chunk| chunk.null_count()).sum::<usize>() == length).then_some(length)
}

/// [`all_null`], or a column that repeats one element it does hold.
pub(super) fn repeated_element_len<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<usize> {
    all_null(ca.chunks(), ca.len()).or_else(|| ca.repeats_one_element().then_some(ca.len()))
}

/// [`repeated_element_len`], for a series whose type is only known at runtime — a nested one,
/// whose `distinct` answers are otherwise read off the groups its rows fall into.
pub(super) fn repeated_element_len_series(s: &Series) -> Option<usize> {
    all_null(s.chunks(), s.len()).or_else(|| s.repeats_one_element().then_some(s.len()))
}

/// A mask of `length` elements that is set at `index` alone.
pub(super) fn only(name: PlSmallStr, length: usize, index: usize) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(length);
    out.extend_constant(length, false);
    out.set(index, true);

    BooleanChunked::from_bitmap(name, out.into())
}
