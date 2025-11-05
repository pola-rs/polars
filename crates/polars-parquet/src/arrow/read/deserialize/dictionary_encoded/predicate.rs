use arrow::bitmap::{Bitmap, BitmapBuilder};

use super::{oob_dict_idx, verify_dict_indices};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode(
    values: HybridRleDecoder<'_>,
    dict_mask: &Bitmap,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    let num_filtered_dict_values = dict_mask.set_bits();

    let expected_pred_true_mask_len = pred_true_mask.len() + values.len();

    // @NOTE: this has to be changed when there are nulls null
    if num_filtered_dict_values == 0 {
        pred_true_mask.extend_constant(values.len(), false);
    } else if num_filtered_dict_values == 1 {
        let needle = dict_mask.leading_zeros();
        decode_single(values, needle as u32, pred_true_mask)?;
    } else {
        decode_multiple(values, dict_mask, pred_true_mask)?;
    }

    assert_eq!(expected_pred_true_mask_len, pred_true_mask.len());

    Ok(())
}

#[inline(never)]
pub fn decode_single(
    mut values: HybridRleDecoder<'_>,
    needle: u32,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    pred_true_mask.reserve(values.len());

    let mut unpacked = [0u32; 32];
    while let Some(chunk) = values.next_chunk()? {
        match chunk {
            HybridRleChunk::Rle(value, size) => {
                pred_true_mask.extend_constant(size, value == needle);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let size = decoder.len();
                let mut chunked = decoder.chunked();

                for _ in 0..size / 32 {
                    let n = chunked.next_into(&mut unpacked).unwrap();
                    debug_assert_eq!(n, 32);

                    let mut is_equal_mask = 0u64;
                    for (i, &v) in unpacked.iter().enumerate() {
                        is_equal_mask |= u64::from(v == needle) << i;
                    }

                    // SAFETY: We reserved enough in the beginning of the function.
                    unsafe { pred_true_mask.push_word_with_len_unchecked(is_equal_mask, 32) };
                }

                if let Some(n) = chunked.next_into(&mut unpacked) {
                    debug_assert_eq!(n, size % 32);

                    let mut is_equal_mask = 0u64;
                    for (i, &v) in unpacked[..n].iter().enumerate() {
                        is_equal_mask |= u64::from(v == needle) << i;
                    }

                    // SAFETY: We reserved enough in the beginning of the function.
                    unsafe { pred_true_mask.push_word_with_len_unchecked(is_equal_mask, n) };
                }
            },
        }
    }

    Ok(())
}

#[inline(never)]
pub fn decode_multiple(
    mut values: HybridRleDecoder<'_>,
    dict_mask: &Bitmap,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    pred_true_mask.reserve(values.len());

    let mut unpacked = [0u32; 32];
    while let Some(chunk) = values.next_chunk()? {
        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let is_pred_true = dict_mask.get(value as usize).ok_or_else(oob_dict_idx)?;
                pred_true_mask.extend_constant(size, is_pred_true);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let size = decoder.len();
                let mut chunked = decoder.chunked();

                for _ in 0..size / 32 {
                    let n = chunked.next_into(&mut unpacked).unwrap();
                    debug_assert_eq!(n, 32);

                    verify_dict_indices(&unpacked, dict_mask.len())?;
                    let mut is_pred_true_mask = 0u64;
                    for (i, &v) in unpacked.iter().enumerate() {
                        // SAFETY: We just verified the dictionary indices
                        let is_pred_true = unsafe { dict_mask.get_bit_unchecked(v as usize) };
                        is_pred_true_mask |= u64::from(is_pred_true) << i;
                    }

                    // SAFETY: We reserved enough in the beginning of the function.
                    unsafe { pred_true_mask.push_word_with_len_unchecked(is_pred_true_mask, 32) };
                }

                if let Some(n) = chunked.next_into(&mut unpacked) {
                    debug_assert_eq!(n, size % 32);

                    verify_dict_indices(&unpacked[..n], dict_mask.len())?;
                    let mut is_pred_true_mask = 0u64;
                    for (i, &v) in unpacked[..n].iter().enumerate() {
                        // SAFETY: We just verified the dictionary indices
                        let is_pred_true = unsafe { dict_mask.get_bit_unchecked(v as usize) };
                        is_pred_true_mask |= u64::from(is_pred_true) << i;
                    }

                    // SAFETY: We reserved enough in the beginning of the function.
                    unsafe { pred_true_mask.push_word_with_len_unchecked(is_pred_true_mask, n) };
                }
            },
        }
    }

    Ok(())
}
