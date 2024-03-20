use polars_utils::IdxSize;

use crate::array::Array;
use crate::bitmap::Bitmap;
use crate::datatypes::IdxArr;

/// # Safety
/// Doesn't do any bound checks.
pub unsafe fn take_bitmap_unchecked(values: &Bitmap, indices: &[IdxSize]) -> Bitmap {
    let values = indices.iter().map(|&index| {
        debug_assert!((index as usize) < values.len());
        values.get_bit_unchecked(index as usize)
    });
    Bitmap::from_trusted_len_iter(values)
}

/// # Safety
/// Doesn't check bounds for non-null elements.
pub unsafe fn take_bitmap_nulls_unchecked(values: &Bitmap, indices: &IdxArr) -> Bitmap {
    // Fast-path: no need to bother with null indices.
    if indices.null_count() == 0 {
        return take_bitmap_unchecked(values, indices.values());
    }

    if values.is_empty() {
        // Nothing can be in-bounds, assume indices is full-null.
        debug_assert!(indices.null_count() == indices.len());
        return Bitmap::new_zeroed(indices.len());
    }

    let values = indices.iter().map(|opt_index| {
        // We checked that values.len() > 0 so we can use index 0 for nulls.
        let index = opt_index.copied().unwrap_or(0) as usize;
        debug_assert!(index < values.len());
        values.get_bit_unchecked(index)
    });
    Bitmap::from_trusted_len_iter(values)
}
