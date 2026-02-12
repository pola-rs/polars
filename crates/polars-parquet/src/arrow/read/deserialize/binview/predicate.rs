//! Specialized kernels to do predicate evaluation directly on the `BinView` Parquet data.

use arrow::array::View;
use arrow::bitmap::BitmapBuilder;
use polars_utils::aliases::PlIndexSet;

use crate::parquet::error::ParquetResult;

/// Create a mask for when a value is equal to the `needle`.
pub fn decode_equals(
    num_expected_values: usize,
    values: &[u8],
    needle: &[u8],
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    if needle.len() <= View::MAX_INLINE_SIZE as usize {
        let needle_view = View::new_inline(needle);
        decode_matches(
            num_expected_values,
            values,
            |value| {
                value.len() == needle.len()
                    && unsafe { View::new_inline_unchecked(value) } == needle_view
            },
            pred_true_mask,
        )
    } else {
        decode_matches(
            num_expected_values,
            values,
            |value| value == needle,
            pred_true_mask,
        )
    }
}

pub fn decode_is_in_no_values_non_inlinable(
    num_expected_values: usize,
    values: &[u8],
    needles: &PlIndexSet<Box<[u8]>>,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    decode_matches(
        num_expected_values,
        values,
        |value| needles.contains(value),
        pred_true_mask,
    )
}

pub fn decode_is_in_non_inlinable(
    num_expected_values: usize,
    mut values: &[u8],
    needles: &PlIndexSet<Box<[u8]>>,
    needle_views: &[View],
    target: &mut Vec<View>,
    total_bytes_len: &mut usize,
) -> ParquetResult<()> {
    assert_eq!(needles.len(), needle_views.len());
    assert!(!needles.is_empty());

    target.reserve(num_expected_values);
    let mut next_idx = target.len();
    for _ in 0..num_expected_values {
        if values.len() < 4 {
            return Err(super::invalid_input_err());
        }

        let length;
        (length, values) = values.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if values.len() < length as usize {
            return Err(super::invalid_input_err());
        }

        let value;
        (value, values) = values.split_at(length as usize);

        let needle_idx = needles.get_index_of(value);
        unsafe {
            // SAFETY: We checked that 0 < needle_views.len() == needles.len().
            let view = needle_views.get_unchecked(needle_idx.unwrap_or_default());

            // SAFETY: We reserved enough just before the loop.
            target.as_mut_ptr().add(next_idx).write(*view);
        }

        if needle_idx.is_some() {
            *total_bytes_len += value.len();
        }
        next_idx += usize::from(needle_idx.is_some());
    }

    // SAFETY: We wrote all these items. Note, that views are Copy, so erroring or panicked until
    // this point won't miss Drop calls.
    unsafe {
        target.set_len(next_idx);
    }

    Ok(())
}

pub fn decode_is_in_no_values_inlinable(
    num_expected_values: usize,
    values: &[u8],
    needles: &[View; 4],
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    decode_matches(
        num_expected_values,
        values,
        |value| {
            let length = value.len() as u32;
            if length > View::MAX_INLINE_SIZE {
                return false;
            }
            // SAFETY: we made sure length <= View::MAX_INLINE_SIZE.
            let mut view = unsafe { View::new_inline_unchecked(value) };
            view.length = length;
            needles.contains(&view)
        },
        pred_true_mask,
    )
}

pub fn decode_is_in_inlinable(
    num_expected_values: usize,
    mut values: &[u8],
    needles: &[View; 4],
    target: &mut Vec<View>,
    total_bytes_len: &mut usize,
) -> ParquetResult<()> {
    target.reserve(num_expected_values);
    let mut next_idx = target.len();
    for _ in 0..num_expected_values {
        if values.len() < 4 {
            return Err(super::invalid_input_err());
        }

        let length;
        (length, values) = values.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if values.len() < length as usize {
            return Err(super::invalid_input_err());
        }

        // Always advance the slice before checking length.
        let value;
        (value, values) = values.split_at(length as usize);

        // Non-inlinable views can't match inlinable needles.
        if length > View::MAX_INLINE_SIZE {
            continue;
        }
        // SAFETY: we made sure length <= View::MAX_INLINE_SIZE.
        let mut view = unsafe { View::new_inline_unchecked(value) };
        view.length = length;

        let is_pred_true = needles.contains(&view);
        // SAFETY: We reserved enough just before the loop.
        unsafe {
            target.as_mut_ptr().add(next_idx).write(view);
        }
        if is_pred_true {
            *total_bytes_len += value.len();
        }
        next_idx += usize::from(is_pred_true);
    }

    // SAFETY: We wrote all these items. Note, that views are Copy, so erroring or panicked until
    // this point won't miss Drop calls.
    unsafe {
        target.set_len(next_idx);
    }

    Ok(())
}

pub fn decode_matches(
    num_expected_values: usize,
    mut values: &[u8],
    is_match: impl Fn(&[u8]) -> bool,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    pred_true_mask.reserve(num_expected_values);
    for _ in 0..num_expected_values {
        if values.len() < 4 {
            return Err(super::invalid_input_err());
        }

        let length;
        (length, values) = values.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if values.len() < length as usize {
            return Err(super::invalid_input_err());
        }

        let value;
        (value, values) = values.split_at(length as usize);

        let is_match = is_match(value);
        // SAFETY: We reserved enough just before the loop.
        unsafe { pred_true_mask.push_unchecked(is_match) };
    }

    Ok(())
}
