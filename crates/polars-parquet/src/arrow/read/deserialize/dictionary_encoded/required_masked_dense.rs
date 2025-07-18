use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::AlignedBytes;

use super::{IndexMapping, oob_dict_idx, required_skip_whole_chunks, verify_dict_indices};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    mut filter: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    assert!(values.len() >= filter.len());

    let mut num_rows_to_skip = filter.take_leading_zeros();
    filter.take_trailing_zeros();

    let num_rows = filter.set_bits();

    values.limit_to(num_rows_to_skip + filter.len());

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        return super::required::decode(values, dict, target, num_rows_to_skip);
    }

    if dict.is_empty() && !filter.is_empty() {
        return Err(oob_dict_idx());
    }

    target.reserve(num_rows);

    let mut filter = BitMask::from_bitmap(&filter);

    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    // Skip over whole HybridRleChunks
    required_skip_whole_chunks(&mut values, &mut num_rows_to_skip)?;

    while let Some(chunk) = values.next_chunk()? {
        debug_assert!(num_rows_to_skip < chunk.len() || chunk.len() == 0);

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

                let current_filter;

                (current_filter, filter) = filter.split_at(size - num_rows_to_skip);
                let num_chunk_rows = current_filter.set_bits();

                if num_chunk_rows > 0 {
                    let Some(value) = dict.get(value) else {
                        return Err(oob_dict_idx());
                    };

                    target.resize(target.len() + num_chunk_rows, value);
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let size = decoder.len();
                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;
                let mut skip_values = num_rows_to_skip;

                let current_filter;

                (current_filter, filter) = filter.split_at(size - num_rows_to_skip);

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
                    let target_ptr = unsafe { target.as_mut_ptr().add(target.len()) };

                    while f != 0 {
                        let offset = f.trailing_zeros() as usize;

                        num_read += offset;

                        let idx = values_buffer[(values_offset + num_read) % 128];
                        // SAFETY:
                        // 1. `values_buffer` starts out as only zeros, which we know is in the
                        //    dictionary following the original `dict.is_empty` check.
                        // 2. Each time we write to `values_buffer`, it is followed by a
                        //    `verify_dict_indices`.
                        let value = unsafe { dict.get_unchecked(idx) };
                        unsafe { target_ptr.add(num_written).write(value) };

                        num_written += 1;
                        num_read += 1;

                        f >>= offset + 1; // Clear least significant bit.
                    }

                    values_offset += len;
                    values_offset %= 128;
                    num_buffered -= len;
                    unsafe {
                        target.set_len(target.len() + num_written);
                    }

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

        num_rows_to_skip = 0;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::proptest::bitmap;
    use arrow::types::Bytes4Alignment4;
    use proptest::collection::size_range;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    use super::*;
    use crate::parquet::encoding::hybrid_rle;

    fn values_and_mask() -> impl Strategy<Value = (Vec<u8>, u32, Vec<u32>, Bitmap)> {
        (bitmap(1..100), (0..300u32)).prop_flat_map(|(mask, max_idx)| {
            let len = mask.len();
            let hybrid_rle = hybrid_rle::proptest::hybrid_rle(max_idx, len);

            (
                hybrid_rle,
                Just(max_idx),
                any_with::<Vec<u32>>(size_range((max_idx + 1) as usize).lift()),
                Just(mask),
            )
        })
    }

    fn _test_decode(
        hybrid_rle: HybridRleDecoder<'_>,
        dict: &[Bytes4Alignment4],
        mask: &Bitmap,
    ) -> TestCaseResult {
        let mut result = Vec::<arrow::types::Bytes4Alignment4>::with_capacity(mask.set_bits());
        decode(hybrid_rle.clone(), dict, mask.clone(), &mut result).unwrap();

        let idxs = hybrid_rle.collect().unwrap();
        let mut result_i = 0;
        for (idx, is_selected) in idxs.iter().zip(mask.iter()) {
            if is_selected {
                prop_assert_eq!(result[result_i], dict[*idx as usize]);
                result_i += 1;
            }
        }

        TestCaseResult::Ok(())
    }

    proptest! {
        #[test]
        fn test_decode_masked_optional(
            (ref hybrid_rle, max_idx, ref dict, ref mask) in values_and_mask()
        ) {
            let hybrid_rle = HybridRleDecoder::new(hybrid_rle, 32 - max_idx.leading_zeros(), mask.len());
            let dict = bytemuck::cast_slice(dict.as_slice());
            _test_decode(hybrid_rle, dict, mask)?
        }
    }
}
