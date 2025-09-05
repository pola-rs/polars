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

#[inline(never)]
pub fn decode_is_in_no_values<B: AlignedBytes>(
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

#[inline(never)]
pub fn decode_is_in<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    needle: &[B; 8],
    target: &mut Vec<B>,
    pred_true_mask: &mut BitmapBuilder,
) {
    target.reserve(values.len());
    pred_true_mask.reserve(values.len());
    let mut next_idx = target.len();
    for &v in values {
        let v = B::from_unaligned(v);
        let is_pred_true = needle.contains(&v);

        // SAFETY: We reserved enough before the loop.
        unsafe {
            target.as_mut_ptr().add(next_idx).write(v);
            pred_true_mask.push_unchecked(is_pred_true);
        }

        next_idx += usize::from(is_pred_true);
    }
    unsafe {
        target.set_len(next_idx);
    }
}
