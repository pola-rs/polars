//! What the `distinct` family answers of a chunked array whose one chunk repeats a single element.

use arrow::bitmap::MutableBitmap;
use polars_core::prelude::*;

/// The length of `ca`, if its one chunk repeats a single element more than once.
///
/// Every element of such a chunk is the same one — the same value throughout, or a null
/// throughout — which settles the whole `distinct` family on it without a single element being
/// hashed: the first element is the only one that is distinct in it, so is the last, and none of
/// them occurs just once.
pub(super) fn repeated_element_len<T: PolarsDataType>(ca: &ChunkedArray<T>) -> Option<usize> {
    let [chunk] = ca.chunks().as_slice() else {
        return None;
    };

    (ca.len() > 1 && chunk.is_scalar()).then(|| ca.len())
}

/// A mask of `length` elements that is set at `index` alone.
pub(super) fn only(name: PlSmallStr, length: usize, index: usize) -> BooleanChunked {
    let mut out = MutableBitmap::with_capacity(length);
    out.extend_constant(length, false);
    out.set(index, true);

    BooleanChunked::from_bitmap(name, out.into())
}
