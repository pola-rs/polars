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
    validity: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    let num_valid_values = validity.set_bits();

    // Dispatch to the required kernel if all rows are valid anyway.
    if num_valid_values == validity.len() {
        values.limit_to(validity.len());
        return super::required::decode(values, dict, target);
    }

    if dict.is_empty() && num_valid_values > 0 {
        return Err(oob_dict_idx());
    }

    assert!(num_valid_values <= values.len());
    let start_length = target.len();
    let end_length = start_length + validity.len();

    target.reserve(validity.len());
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    values.limit_to(num_valid_values);
    let mut validity = BitMask::from_bitmap(&validity);
    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    for chunk in values.into_chunk_iter() {
        match chunk? {
            HybridRleChunk::Rle(value, size) => {
                if size == 0 {
                    continue;
                }

                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let num_chunk_rows = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

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
                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;

                {
                    let mut num_done = 0;
                    let mut validity_iter = validity.fast_iter_u56();

                    'outer: for v in validity_iter.by_ref() {
                        while num_buffered < v.count_ones() as usize {
                            let buffer_part = <&mut [u32; 32]>::try_from(
                                &mut values_buffer[buffer_part_idx * 32..][..32],
                            )
                            .unwrap();
                            let Some(num_added) = chunked.next_into(buffer_part) else {
                                break 'outer;
                            };

                            verify_dict_indices(buffer_part, dict.len())?;

                            num_buffered += num_added;

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
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity.set_bits(), 0);
    }

    let target_slice = unsafe { std::slice::from_raw_parts_mut(target_ptr, validity.len()) };
    target_slice.fill(B::zeroed());
    unsafe {
        target.set_len(end_length);
    }

    Ok(())
}
