//! What the `distinct` family answers of a chunked array whose one chunk repeats a single element.

use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;

/// `length`, if every one of the `length` elements `chunks` holds is the same one.
fn all_null(chunks: &[PlArrayRef], length: usize) -> Option<usize> {
    if length <= 1 {
        return None;
    }

    (chunks.iter().map(|chunk| chunk.null_count()).sum::<usize>() == length).then_some(length)
}

/// [`all_null`], or a column that repeats one element it does hold.
pub(super) fn repeated_element_len<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<usize> {
    all_null(ca.chunks(), ca.len()).or_else(|| ca.repeats_one_element().then_some(ca.len()))
}

/// [`repeated_element_len`], for a series whose type is only known at runtime.
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
