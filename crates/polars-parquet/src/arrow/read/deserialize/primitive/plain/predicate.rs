use arrow::bitmap::BitmapBuilder;
use arrow::types::AlignedBytes;

use super::ArrayChunks;

#[inline(never)]
pub fn decode_equals<B: AlignedBytes>(
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

#[inline(never)]
pub fn decode_between<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    low1: B,
    high1: B,
    low2: B,
    high2: B,
    pred_true_mask: &mut BitmapBuilder,
) {
    pred_true_mask.reserve(values.len());
    for &v in values {
        let v = B::from_unaligned(v);
        let in_interval1 = (low1.unsigned_leq(v)) & (v.unsigned_leq(high1));
        let in_interval2 = (low2.unsigned_leq(v)) & (v.unsigned_leq(high2));

        let is_pred_true = in_interval1 | in_interval2;

        // SAFETY: We reserved enough before the loop.
        unsafe {
            pred_true_mask.push_unchecked(is_pred_true);
        }
    }
}

#[inline(never)]
pub fn decode_is_in<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    needle: &[B; 8],
    pred_true_mask: &mut BitmapBuilder,
) {
    pred_true_mask.reserve(values.len());
    for &v in values {
        let is_pred_true = needle.contains(&B::from_unaligned(v));

        // SAFETY: We reserved enough before the loop.
        unsafe {
            pred_true_mask.push_unchecked(is_pred_true);
        }
    }
}
