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
    filter: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    let num_rows = filter.set_bits();

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        values.limit_to(filter.len());
        return super::required::decode(values, dict, target, 0);
    }

    if dict.is_empty() && !filter.is_empty() {
        return Err(oob_dict_idx());
    }

    let start_length = target.len();

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut filter = BitMask::from_bitmap(&filter);

    values.limit_to(filter.len());
    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    let mut num_rows_left = num_rows;

    for chunk in values.into_chunk_iter() {
        if num_rows_left == 0 {
            break;
        }

        match chunk? {
            HybridRleChunk::Rle(value, size) => {
                if size == 0 {
                    continue;
                }

                let size = size.min(filter.len());

                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let current_filter;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(size) };
                let num_chunk_rows = current_filter.set_bits();

                if num_chunk_rows > 0 {
                    let target_slice;
                    // SAFETY:
                    // Given `filter_iter` before the `advance_by_bits`.
                    //
                    // 1. `target_ptr..target_ptr + filter_iter.count_ones()` is allocated
                    // 2. `num_chunk_rows < filter_iter.count_ones()`
                    unsafe {
                        target_slice = std::slice::from_raw_parts_mut(target_ptr, num_chunk_rows);
                        target_ptr = target_ptr.add(num_chunk_rows);
                    }

                    let Some(value) = dict.get(value as usize) else {
                        return Err(oob_dict_idx());
                    };

                    target_slice.fill(*value);
                    num_rows_left -= num_chunk_rows;
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let size = decoder.len().min(filter.len());
                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;
                let mut skip_values = 0;

                let current_filter;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(size) };

                let mut iter = |mut f: u64, len: usize| {
                    debug_assert!(len <= 64);

                    // Skip chunk if we don't any values from here.
                    if f == 0 {
                        skip_values += len;
                        return ParquetResult::Ok(());
                    }

                    // Skip over already buffered items.
                    let num_buffered_skipped = skip_values.min(num_buffered);
                    values_offset += num_buffered_skipped;
                    num_buffered -= num_buffered_skipped;
                    skip_values -= num_buffered_skipped;

                    // If we skipped plenty already, just skip decoding those chunks instead of
                    // decoding them and throwing them away.
                    chunked.decoder.skip_chunks(skip_values / 32);
                    // The leftovers we have to decode but we can also just skip.
                    skip_values %= 32;

                    while num_buffered < len {
                        let buffer_part = <&mut [u32; 32]>::try_from(
                            &mut values_buffer[buffer_part_idx * 32..][..32],
                        )
                        .unwrap();
                        let num_added = chunked.next_into(buffer_part).unwrap();

                        verify_dict_indices(buffer_part, dict.len())?;

                        let skip_chunk_values = skip_values.min(num_added);

                        values_offset += skip_chunk_values;
                        num_buffered += num_added - skip_chunk_values;
                        skip_values -= skip_chunk_values;

                        buffer_part_idx += 1;
                        buffer_part_idx %= 4;
                    }

                    let mut num_read = 0;
                    let mut num_written = 0;

                    while f != 0 {
                        let offset = f.trailing_zeros() as usize;

                        num_read += offset;

                        let idx = values_buffer[(values_offset + num_read) % 128];
                        // SAFETY:
                        // 1. `values_buffer` starts out as only zeros, which we know is in the
                        //    dictionary following the original `dict.is_empty` check.
                        // 2. Each time we write to `values_buffer`, it is followed by a
                        //    `verify_dict_indices`.
                        let value = *unsafe { dict.get_unchecked(idx as usize) };
                        unsafe { target_ptr.add(num_written).write(value) };

                        num_written += 1;
                        num_read += 1;

                        f >>= offset + 1; // Clear least significant bit.
                    }

                    values_offset += len;
                    values_offset %= 128;
                    num_buffered -= len;
                    unsafe {
                        target_ptr = target_ptr.add(num_written);
                    }
                    num_rows_left -= num_written;

                    ParquetResult::Ok(())
                };

                let mut f_iter = current_filter.fast_iter_u56();

                for f in f_iter.by_ref() {
                    iter(f, 56)?;
                }

                let (f, fl) = f_iter.remainder();

                iter(f, fl)?;
            },
        }
    }

    unsafe {
        target.set_len(start_length + num_rows);
    }

    Ok(())
}
