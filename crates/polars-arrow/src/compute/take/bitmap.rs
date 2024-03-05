use polars_utils::IdxSize;

use crate::bitmap::Bitmap;

/// # Safety
/// doesn't do any bound checks
pub unsafe fn take_bitmap_unchecked(values: &Bitmap, indices: &[IdxSize]) -> Bitmap {
    let values = indices.iter().map(|&index| {
        debug_assert!((index as usize) < values.len());
        values.get_bit_unchecked(index as usize)
    });
    Bitmap::from_trusted_len_iter(values)
}
