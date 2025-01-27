use arrow::bitmap::MutableBitmap;
use arrow::types::AlignedBytes;

use super::ArrayChunks;

#[inline(never)]
pub fn decode_equals_no_values<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    needle: B,
    pred_true_mask: &mut MutableBitmap,
) {
    pred_true_mask.extend(values.into_iter().map(|&v| B::from_unaligned(v) == needle))
}
