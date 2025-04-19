use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::AlignedBytes;

use super::{
    IndexMapping, no_more_bitpacked_values, oob_dict_idx, optional_skip_whole_chunks,
    verify_dict_indices,
};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

/// Decoding kernel for optional dictionary encoded.
#[inline(never)]
pub fn decode<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    mut validity: Bitmap,
    target: &mut Vec<B>,
    mut num_rows_to_skip: usize,
) -> ParquetResult<()> {
    debug_assert!(num_rows_to_skip <= validity.len());

    let num_rows = validity.len() - num_rows_to_skip;
    let end_length = target.len() + num_rows;

    target.reserve(num_rows);

    // Remove any leading and trailing nulls. This has two benefits:
    // 1. It increases the chance of dispatching to the faster kernel (e.g. for sorted data)
    // 2. It reduces the amount of iterations in the main loop and replaces it with `memset`s
    let leading_nulls = validity.take_leading_zeros();
    let trailing_nulls = validity.take_trailing_zeros();

    // Special case: all values are skipped, just add the trailing null.
    if num_rows_to_skip >= leading_nulls + validity.len() {
        target.resize(end_length, B::zeroed());
        return Ok(());
    }

    values.limit_to(validity.set_bits());

    // Add the leading nulls
    if num_rows_to_skip < leading_nulls {
        target.resize(target.len() + leading_nulls - num_rows_to_skip, B::zeroed());
        num_rows_to_skip = 0;
    } else {
        num_rows_to_skip -= leading_nulls;
    }

    if validity.set_bits() == validity.len() {
        // Dispatch to the required kernel if all rows are valid anyway.
        super::required::decode(values, dict, target, num_rows_to_skip)?;
    } else {
        if dict.is_empty() {
            return Err(oob_dict_idx());
        }

        let mut num_values_to_skip = 0;
        if num_rows_to_skip > 0 {
            num_values_to_skip = validity.clone().sliced(0, num_rows_to_skip).set_bits();
        }

        let mut validity = BitMask::from_bitmap(&validity);
        let mut values_buffer = [0u32; 128];
        let values_buffer = &mut values_buffer;

        // Skip over any whole HybridRleChunks
        optional_skip_whole_chunks(
            &mut values,
            &mut validity,
            &mut num_rows_to_skip,
            &mut num_values_to_skip,
        )?;

        while let Some(chunk) = values.next_chunk()? {
            debug_assert!(num_values_to_skip < chunk.len() || chunk.len() == 0);

            match chunk {
                HybridRleChunk::Rle(value, size) => {
                    if size == 0 {
                        continue;
                    }

                    // If we know that we have `size` times `value` that we can append, but there
                    // might be nulls in between those values.
                    //
                    // 1. See how many `num_rows = valid + invalid` values `size` would entail.
                    //    This is done with `nth_set_bit_idx` on the validity mask.
                    // 2. Fill `num_rows` values into the target buffer.
                    // 3. Advance the validity mask by `num_rows` values.

                    let Some(value) = dict.get(value) else {
                        return Err(oob_dict_idx());
                    };

                    let num_chunk_rows =
                        validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());
                    validity.advance_by(num_chunk_rows);

                    target.resize(target.len() + num_chunk_rows - num_rows_to_skip, value);
                },
                HybridRleChunk::Bitpacked(mut decoder) => {
                    let num_rows_for_decoder = validity
                        .nth_set_bit_idx(decoder.len(), 0)
                        .unwrap_or(validity.len());

                    let mut chunked = decoder.chunked();

                    let mut buffer_part_idx = 0;
                    let mut values_offset = 0;
                    let mut num_buffered: usize = 0;

                    let mut decoder_validity;
                    (decoder_validity, validity) = validity.split_at(num_rows_for_decoder);

                    // Skip over any remaining values.
                    if num_rows_to_skip > 0 {
                        decoder_validity.advance_by(num_rows_to_skip);

                        chunked.decoder.skip_chunks(num_values_to_skip / 32);
                        num_values_to_skip %= 32;

                        if num_values_to_skip > 0 {
                            let buffer_part = <&mut [u32; 32]>::try_from(
                                &mut values_buffer[buffer_part_idx * 32..][..32],
                            )
                            .unwrap();
                            let Some(num_added) = chunked.next_into(buffer_part) else {
                                return Err(no_more_bitpacked_values());
                            };

                            debug_assert!(num_values_to_skip <= num_added);
                            verify_dict_indices(buffer_part, dict.len())?;

                            values_offset += num_values_to_skip;
                            num_buffered += num_added - num_values_to_skip;
                            buffer_part_idx += 1;
                        }
                    }

                    let mut iter = |v: u64, n: usize| {
                        while num_buffered < v.count_ones() as usize {
                            buffer_part_idx %= 4;

                            let buffer_part = <&mut [u32; 32]>::try_from(
                                &mut values_buffer[buffer_part_idx * 32..][..32],
                            )
                            .unwrap();
                            let Some(num_added) = chunked.next_into(buffer_part) else {
                                return Err(no_more_bitpacked_values());
                            };

                            verify_dict_indices(buffer_part, dict.len())?;

                            num_buffered += num_added;

                            buffer_part_idx += 1;
                        }

                        let mut num_read = 0;

                        target.extend((0..n).map(|i| {
                            let idx = values_buffer[(values_offset + num_read) % 128];
                            num_read += ((v >> i) & 1) as usize;

                            // SAFETY:
                            // 1. `values_buffer` starts out as only zeros, which we know is in the
                            //    dictionary following the original `dict.is_empty` check.
                            // 2. Each time we write to `values_buffer`, it is followed by a
                            //    `verify_dict_indices`.
                            unsafe { dict.get_unchecked(idx) }
                        }));

                        values_offset += num_read;
                        values_offset %= 128;
                        num_buffered -= num_read;

                        ParquetResult::Ok(())
                    };

                    let mut v_iter = decoder_validity.fast_iter_u56();
                    for v in v_iter.by_ref() {
                        iter(v, 56)?;
                    }

                    let (v, vl) = v_iter.remainder();
                    iter(v, vl)?;
                },
            }

            num_rows_to_skip = 0;
            num_values_to_skip = 0;
        }
    }

    // Add back the trailing nulls
    debug_assert_eq!(target.len(), end_length - trailing_nulls);
    target.resize(end_length, B::zeroed());

    Ok(())
}
