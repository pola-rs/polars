use arrow::bitmap::BitmapBuilder;
use arrow::types::AlignedBytes;

use super::ArrayChunks;

#[inline(never)]
pub fn decode_equals_no_values<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    needle: B,
    pred_true_mask: &mut BitmapBuilder,
) {
    pred_true_mask.reserve(values.len());
    for &v in values {
        let is_pred_true = B::from_unaligned(v) == needle;

        // SAFETY: We reserved enough before the loop.
        unsafe {
            pred_true_mask.push_unchecked(is_pred_true);
        }
    }
}
