use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::AlignedBytes;

use super::{oob_dict_idx, verify_dict_indices, IndexMapping};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;
use crate::read::PredicateFilter;

#[inline(never)]
pub fn decode<B: AlignedBytes, D: IndexMapping<Output = B>>(
    values: HybridRleDecoder<'_>,
    dict: D,
    dict_mask: &Bitmap,
    predicate: &PredicateFilter,
    target: &mut Vec<B>,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    let num_filtered_dict_values = dict_mask.set_bits();

    let expected_pred_true_mask_len = pred_true_mask.len() + values.len();

    // @NOTE: this has to be changed when there are nulls null
    if num_filtered_dict_values == 0 {
        pred_true_mask.extend_constant(values.len(), false);
    } else if num_filtered_dict_values == 1 {
        let needle = dict_mask.leading_zeros();
        let start_num_values = pred_true_mask.set_bits();

        decode_single_no_values(values, needle as u32, pred_true_mask)?;

        if predicate.include_values {
            let num_values = pred_true_mask.set_bits() - start_num_values;
            target.resize(target.len() + num_values, dict.get(needle as u32).unwrap());
        }
    } else if predicate.include_values {
        decode_multiple_values(values, dict, dict_mask, target, pred_true_mask)?;
    } else {
        decode_multiple_no_values(values, dict_mask, pred_true_mask)?;
    }

    assert_eq!(expected_pred_true_mask_len, pred_true_mask.len());

    Ok(())
}

#[inline(never)]
pub fn decode_single_no_values(
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
                    for (i, &v) in unpacked.iter().enumerate() {
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
pub fn decode_multiple_no_values(
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

#[inline(never)]
pub fn decode_multiple_values<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    dict_mask: &Bitmap,
    target: &mut Vec<B>,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    pred_true_mask.reserve(values.len());
    target.reserve(values.len());

    let mut unpacked = [0u32; 32];
    while let Some(chunk) = values.next_chunk()? {
        match chunk {
            HybridRleChunk::Rle(value, size) => {
                if size == 0 {
                    continue;
                }

                let is_pred_true = dict_mask.get(value as usize).ok_or_else(oob_dict_idx)?;
                pred_true_mask.extend_constant(size, is_pred_true);
                if is_pred_true {
                    let value = dict.get(value).unwrap();
                    target.resize(target.len() + size, value);
                }
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

                    let num_pred_true_values = is_pred_true_mask.count_ones() as usize;

                    if num_pred_true_values == 32 {
                        target.extend(
                            unpacked
                                .iter()
                                // SAFETY: We just verified the dictionary indices
                                .map(|&v| unsafe { dict.get_unchecked(v) }),
                        );
                    } else if num_pred_true_values > 0 {
                        let mut write_ptr = unsafe { target.as_mut_ptr().add(target.len()) };
                        for v in unpacked {
                            unsafe {
                                write_ptr.write(dict.get_unchecked(v));
                                let select = dict_mask.get_bit_unchecked(v as usize);
                                write_ptr = write_ptr.add(usize::from(select));
                            }
                        }

                        let new_len = target.len() + num_pred_true_values;
                        unsafe { target.set_len(new_len) };
                    }
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

                    let num_pred_true_values = is_pred_true_mask.count_ones() as usize;

                    if num_pred_true_values == n {
                        target.extend(
                            unpacked[..n]
                                .iter()
                                // SAFETY: We just verified the dictionary indices
                                .map(|&v| unsafe { dict.get_unchecked(v) }),
                        );
                    } else if num_pred_true_values > 0 {
                        let mut write_ptr = unsafe { target.as_mut_ptr().add(target.len()) };
                        for &v in &unpacked[..n] {
                            unsafe {
                                write_ptr.write(dict.get_unchecked(v));
                                let select = dict_mask.get_bit_unchecked(v as usize);
                                write_ptr = write_ptr.add(usize::from(select));
                            }
                        }

                        let new_len = target.len() + num_pred_true_values;
                        unsafe { target.set_len(new_len) };
                    }
                }
            },
        }
    }

    Ok(())
}
