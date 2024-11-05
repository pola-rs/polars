use arrow::array::Splitable;
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
    mut filter: Bitmap,
    mut validity: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    let leading_filtered = filter.take_leading_zeros();
    filter.take_trailing_zeros();

    let num_rows = filter.set_bits();

    let leading_validity;
    (leading_validity, validity) = validity.split_at(leading_filtered);

    let mut num_rows_to_skip = leading_filtered;
    let mut num_values_to_skip = leading_validity.set_bits();

    validity = validity.sliced(0, filter.len());

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        return super::optional::decode(values, dict, validity, target, num_rows_to_skip);
    }
    // Dispatch to the required kernel if all rows are valid anyway.
    if validity.set_bits() == validity.len() {
        return super::required_masked_dense::decode(
            values,
            dict,
            filter,
            target,
            num_values_to_skip,
        );
    }
    if dict.is_empty() && validity.set_bits() > 0 {
        return Err(oob_dict_idx());
    }

    debug_assert_eq!(filter.len(), validity.len());
    assert!(validity.set_bits() <= values.len());
    let start_length = target.len();

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut filter = BitMask::from_bitmap(&filter);
    let mut validity = BitMask::from_bitmap(&validity);

    values.limit_to(num_values_to_skip + validity.set_bits());
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

            if num_values_to_skip < chunk_len {
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
        match chunk {
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

                let num_chunk_values = validity
                    .nth_set_bit_idx(size, num_rows_to_skip)
                    .unwrap_or(validity.len());

                let current_filter;
                (_, validity) = unsafe { validity.split_at_unchecked(num_chunk_values) };
                (current_filter, filter) = unsafe { filter.split_at_unchecked(num_chunk_values) };

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
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                if num_values_to_skip > 0 {
                    validity = validity.sliced(num_rows_to_skip, validity.len() - num_rows_to_skip);
                    decoder.skip_chunks(num_values_to_skip / 32);
                    num_values_to_skip %= 32;
                }

                // For bitpacked we do the following:
                // 1. See how many rows are encoded by this `decoder`.
                // 2. Go through the filter and validity 56 bits at a time and:
                //    0. If filter bits are 0, skip the chunk entirely.
                //    1. Buffer enough values so that we can branchlessly decode with the filter
                //       and validity.
                //    2. Decode with filter and validity.
                // 3. Decode remainder.

                let size = decoder.len();
                let mut chunked = decoder.chunked();

                let num_chunk_values = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;
                let mut skip_values = 0;

                let current_filter;
                let current_validity;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(num_chunk_values) };
                (current_validity, validity) =
                    unsafe { validity.split_at_unchecked(num_chunk_values) };

                let mut iter = |mut f: u64, mut v: u64| {
                    // Skip chunk if we don't any values from here.
                    if f == 0 {
                        skip_values += v.count_ones() as usize;
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

                    while num_buffered < v.count_ones() as usize {
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
                        let offset = f.trailing_zeros();

                        num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                        v >>= offset;

                        let idx = values_buffer[(values_offset + num_read) % 128];
                        // SAFETY:
                        // 1. `values_buffer` starts out as only zeros, which we know is in the
                        //    dictionary following the original `dict.is_empty` check.
                        // 2. Each time we write to `values_buffer`, it is followed by a
                        //    `verify_dict_indices`.
                        let value = unsafe { dict.get_unchecked(idx as usize) };
                        let value = *value;
                        unsafe { target_ptr.add(num_written).write(value) };

                        num_written += 1;
                        num_read += (v & 1) as usize;

                        f >>= offset + 1; // Clear least significant bit.
                        v >>= 1;
                    }

                    num_read += v.count_ones() as usize;

                    values_offset += num_read;
                    values_offset %= 128;
                    num_buffered -= num_read;
                    unsafe {
                        target_ptr = target_ptr.add(num_written);
                    }

                    ParquetResult::Ok(())
                };

                let mut f_iter = current_filter.fast_iter_u56();
                let mut v_iter = current_validity.fast_iter_u56();

                for (f, v) in f_iter.by_ref().zip(v_iter.by_ref()) {
                    iter(f, v)?;
                }

                let (f, fl) = f_iter.remainder();
                let (v, vl) = v_iter.remainder();

                assert_eq!(fl, vl);

                iter(f, v)?;
            },
        }
        
        num_rows_to_skip = 0;
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity.set_bits(), 0);
    }

    let target_slice = unsafe { std::slice::from_raw_parts_mut(target_ptr, validity.len()) };
    target_slice.fill(B::zeroed());
    unsafe {
        target.set_len(start_length + num_rows);
    }

    Ok(())
}
