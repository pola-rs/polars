use arrow::array::View;
use arrow::bitmap::MutableBitmap;

use crate::parquet::error::ParquetResult;

pub fn decode_equals(
    num_expected_values: usize,
    values: &[u8],
    needle: &[u8],
    pred_true_mask: &mut MutableBitmap,
) -> ParquetResult<()> {
    if needle.len() <= View::MAX_INLINE_SIZE as usize {
        decode_equals_inlinable(num_expected_values, values, needle, pred_true_mask)
    } else {
        decode_equals_non_inlineable(num_expected_values, values, needle, pred_true_mask)
    }
}

fn decode_equals_inlinable(
    num_expected_values: usize,
    mut values: &[u8],
    needle: &[u8],
    pred_true_mask: &mut MutableBitmap,
) -> ParquetResult<()> {
    let needle = View::new_inline(needle);

    pred_true_mask.reserve(num_expected_values);

    let expected_pred_true_mask_len = pred_true_mask.len() + num_expected_values;
    pred_true_mask.extend((0..num_expected_values).map_while(|_| {
        if values.len() < 4 {
            return None;
        }

        let length;
        (length, values) = values.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if values.len() < length as usize {
            return None;
        }

        let value;
        (value, values) = values.split_at(length as usize);
        let view = View::new_from_bytes(value, 0, 0);

        Some(needle == view)
    }));

    if expected_pred_true_mask_len != pred_true_mask.len() {
        return Err(super::invalid_input_err());
    }

    Ok(())
}

fn decode_equals_non_inlineable(
    num_expected_values: usize,
    mut values: &[u8],
    needle: &[u8],
    pred_true_mask: &mut MutableBitmap,
) -> ParquetResult<()> {
    pred_true_mask.reserve(num_expected_values);

    let expected_pred_true_mask_len = pred_true_mask.len() + num_expected_values;
    pred_true_mask.extend((0..num_expected_values).map_while(|_| {
        if values.len() < 4 {
            return None;
        }

        let length;
        (length, values) = values.split_at(4);
        let length: &[u8; 4] = unsafe { length.try_into().unwrap_unchecked() };
        let length = u32::from_le_bytes(*length);

        if values.len() < length as usize {
            return None;
        }

        let value;
        (value, values) = values.split_at(length as usize);

        Some(length as usize == needle.len() && value == needle)
    }));

    if expected_pred_true_mask_len != pred_true_mask.len() {
        return Err(super::invalid_input_err());
    }

    Ok(())
}
