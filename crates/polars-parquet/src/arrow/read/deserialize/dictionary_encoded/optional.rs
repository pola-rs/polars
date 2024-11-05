use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;
use arrow::types::AlignedBytes;

use super::{oob_dict_idx, verify_dict_indices};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode<B: AlignedBytes>(
    mut values: HybridRleDecoder<'_>,
    dict: &[B],
    mut validity: Bitmap,
    target: &mut Vec<B>,
    mut num_rows_to_skip: usize,
) -> ParquetResult<()> {
    target.reserve(validity.len() - num_rows_to_skip);

    // Remove any leading and trailing nulls. This has two benefits:
    // 1. It increases the chance of dispatching to the faster kernel (e.g. for sorted data)
    // 2. It reduces the amount of iterations in the main loop and replaces it with `memset`s
    let leading_nulls = validity.take_leading_zeros();
    let trailing_nulls = validity.take_trailing_zeros();

    target.resize(
        target.len() + leading_nulls.saturating_sub(num_rows_to_skip),
        B::zeroed(),
    );
    num_rows_to_skip = num_rows_to_skip.saturating_sub(leading_nulls);

    // Dispatch to the required kernel if all rows are valid anyway.
    if validity.set_bits() == validity.len() {
        values.limit_to(validity.len());
        super::required::decode(values, dict, target, num_rows_to_skip)?;
        target.resize(target.len() + trailing_nulls, B::zeroed());
        return Ok(());
    }
    if dict.is_empty() && validity.set_bits() > 0 {
        return Err(oob_dict_idx());
    }

    let mut num_values_to_skip = validity.clone().sliced(0, num_rows_to_skip).set_bits();

    assert!(validity.set_bits() <= values.len());
    let start_length = target.len();
    let end_length = start_length + validity.len() - num_rows_to_skip;

    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    values.limit_to(validity.set_bits() - num_values_to_skip);
    let mut validity = BitMask::from_bitmap(&validity);
    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    // Skip over any whole HybridRleChunks
    if num_values_to_skip > 0 {
        let mut total_num_skipped_values = 0;

        loop {
            let mut values_clone = values.clone();
            let Some(chunk_len) = values_clone.next_chunk_length()? else {
                break;
            };

            if chunk_len < num_values_to_skip {
                break;
            }

            values = values_clone;
            num_values_to_skip -= chunk_len;
            total_num_skipped_values += chunk_len;
        }

        if total_num_skipped_values > 0 {
            let offset = validity
                .nth_set_bit_idx(total_num_skipped_values - 1, 0)
                .unwrap_or(validity.len());
            num_rows_to_skip -= offset;
            validity = validity.sliced(offset, validity.len() - offset);
        }
    }


    while let Some(chunk) = values.next_chunk()? {
        debug_assert!(chunk.len() < num_rows_to_skip);

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let num_chunk_rows = validity
                    .nth_set_bit_idx(size, num_rows_to_skip)
                    .unwrap_or(validity.len());

                (_, validity) = unsafe { validity.split_at_unchecked(num_chunk_rows) };

                let Some(&value) = dict.get(value as usize) else {
                    return Err(oob_dict_idx());
                };

                let target_slice;
                // SAFETY:
                // Given `validity_iter` before the `advance_by_bits`
                //
                // 1. `target_ptr..target_ptr + validity_iter.bits_left()` is allocated
                // 2. `num_chunk_rows <= validity_iter.bits_left()`
                unsafe {
                    target_slice = std::slice::from_raw_parts_mut(target_ptr, num_chunk_rows);
                    target_ptr = target_ptr.add(num_chunk_rows);
                }

                target_slice.fill(value);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                if num_values_to_skip > 0 {
                    validity = validity.sliced(num_rows_to_skip, validity.len() - num_rows_to_skip);
                    decoder.skip_chunks(num_values_to_skip / 32);
                    num_values_to_skip %= 32;
                }

                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;

                {
                    let mut num_done = 0;
                    let mut validity_iter = validity.fast_iter_u56();

                    'outer: for v in validity_iter.by_ref() {
                        while num_buffered - num_values_to_skip < v.count_ones() as usize {
                            let buffer_part = <&mut [u32; 32]>::try_from(
                                &mut values_buffer[buffer_part_idx * 32..][..32],
                            )
                            .unwrap();
                            let Some(num_added) = chunked.next_into(buffer_part) else {
                                break 'outer;
                            };

                            verify_dict_indices(buffer_part, dict.len())?;

                            let num_added_skipped = num_added.min(num_values_to_skip);
                            num_values_to_skip -= num_added_skipped;

                            num_buffered += num_added - num_added_skipped;

                            buffer_part_idx += 1;
                            buffer_part_idx %= 4;
                        }

                        let mut num_read = 0;

                        for i in 0..56 {
                            let idx = values_buffer[(values_offset + num_read) % 128];

                            // SAFETY:
                            // 1. `values_buffer` starts out as only zeros, which we know is in the
                            //    dictionary following the original `dict.is_empty` check.
                            // 2. Each time we write to `values_buffer`, it is followed by a
                            //    `verify_dict_indices`.
                            let value = unsafe { dict.get_unchecked(idx as usize) };
                            let value = *value;
                            unsafe { target_ptr.add(i).write(value) };
                            num_read += ((v >> i) & 1) as usize;
                        }

                        values_offset += num_read;
                        values_offset %= 128;
                        num_buffered -= num_read;
                        unsafe {
                            target_ptr = target_ptr.add(56);
                        }
                        num_done += 56;
                    }

                    (_, validity) = unsafe { validity.split_at_unchecked(num_done) };
                }

                let num_decoder_remaining = num_buffered + chunked.decoder.len();
                let decoder_limit = validity
                    .nth_set_bit_idx(num_decoder_remaining, 0)
                    .unwrap_or(validity.len());

                let current_validity;
                (current_validity, validity) =
                    unsafe { validity.split_at_unchecked(decoder_limit) };
                let (v, _) = current_validity.fast_iter_u56().remainder();

                while num_buffered < v.count_ones() as usize {
                    let buffer_part = <&mut [u32; 32]>::try_from(
                        &mut values_buffer[buffer_part_idx * 32..][..32],
                    )
                    .unwrap();
                    let num_added = chunked.next_into(buffer_part).unwrap();

                    verify_dict_indices(buffer_part, dict.len())?;

                    num_buffered += num_added;

                    buffer_part_idx += 1;
                    buffer_part_idx %= 4;
                }

                let mut num_read = 0;

                for i in 0..decoder_limit {
                    let idx = values_buffer[(values_offset + num_read) % 128];
                    let value = unsafe { dict.get_unchecked(idx as usize) };
                    let value = *value;
                    unsafe { *target_ptr.add(i) = value };
                    num_read += ((v >> i) & 1) as usize;
                }

                unsafe {
                    target_ptr = target_ptr.add(decoder_limit);
                }
            },
        }

        num_rows_to_skip = 0;
        num_values_to_skip = 0;
    }

    unsafe {
        target.set_len(end_length);
    }
    target.resize(target.len() + trailing_nulls, B::zeroed());

    Ok(())
}
