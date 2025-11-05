use arrow::bitmap::Bitmap;
use arrow::bitmap::bitmask::BitMask;
use arrow::types::AlignedBytes;

use super::{IndexMapping, oob_dict_idx, verify_dict_indices};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    mut filter: Bitmap,
    mut validity: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    // @NOTE: We don't skip leading filtered values, because it is a bit more involved than the
    // other kernels. We could probably do it anyway after having tried to dispatch to faster
    // kernels, but we lose quite a bit of the potency with that.
    filter.take_trailing_zeros();
    validity = validity.sliced(0, filter.len());

    let num_rows = filter.set_bits();
    let num_valid_values = validity.set_bits();

    assert_eq!(filter.len(), validity.len());
    assert!(num_valid_values <= values.len());

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        return super::optional::decode(values, dict, validity, target, 0);
    }

    // Dispatch to the required kernel if all rows are valid anyway.
    if num_valid_values == validity.len() {
        return super::required_masked_dense::decode(values, dict, filter, target);
    }

    if dict.is_empty() && num_valid_values > 0 {
        return Err(oob_dict_idx());
    }

    target.reserve(num_rows);

    let end_length = target.len() + num_rows;

    let mut filter = BitMask::from_bitmap(&filter);
    let mut validity = BitMask::from_bitmap(&validity);

    values.limit_to(num_valid_values);
    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    let mut num_rows_left = num_rows;

    for chunk in values.into_chunk_iter() {
        // Early stop if we have no more rows to load.
        if num_rows_left == 0 {
            break;
        }

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

                let num_chunk_values = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

                let current_filter;
                (current_filter, filter) = filter.split_at(num_chunk_values);
                validity.advance_by(num_chunk_values);

                let num_chunk_rows = current_filter.set_bits();

                let Some(value) = dict.get(value) else {
                    return Err(oob_dict_idx());
                };

                target.resize(target.len() + num_chunk_rows, value);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
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
                    let target_ptr = unsafe { target.as_mut_ptr().add(target.len()) };

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
                        let value = unsafe { dict.get_unchecked(idx) };
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
                        target.set_len(target.len() + num_written);
                    }
                    num_rows_left -= num_written;

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
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity.set_bits(), 0);
    }

    target.resize(end_length, B::zeroed());

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

    fn validity_values_and_mask() -> impl Strategy<Value = (Bitmap, Vec<u8>, u32, Vec<u32>, Bitmap)>
    {
        (bitmap(1..100), (0..300u32)).prop_flat_map(|(validity, max_idx)| {
            let len = validity.len();
            let values_length = validity.set_bits();
            let hybrid_rle = hybrid_rle::proptest::hybrid_rle(max_idx, values_length);

            (
                Just(validity),
                hybrid_rle,
                Just(max_idx),
                any_with::<Vec<u32>>(size_range((max_idx + 1) as usize).lift()),
                bitmap(len),
            )
        })
    }

    fn _test_decode(
        validity: &Bitmap,
        hybrid_rle: HybridRleDecoder<'_>,
        dict: &[Bytes4Alignment4],
        mask: &Bitmap,
    ) -> TestCaseResult {
        let mut result = Vec::<arrow::types::Bytes4Alignment4>::with_capacity(mask.set_bits());
        decode(
            hybrid_rle.clone(),
            dict,
            mask.clone(),
            validity.clone(),
            &mut result,
        )
        .unwrap();

        let idxs = hybrid_rle.collect().unwrap();
        let mut result_i = 0;
        let mut values_i = 0;
        for (is_valid, is_selected) in validity.iter().zip(mask.iter()) {
            if is_selected {
                if is_valid {
                    prop_assert_eq!(result[result_i], dict[idxs[values_i] as usize]);
                }
                result_i += 1;
            }

            if is_valid {
                values_i += 1;
            }
        }

        TestCaseResult::Ok(())
    }

    proptest! {
        #[test]
        fn test_decode_masked_optional(
            (ref validity, ref hybrid_rle, max_idx, ref dict, ref mask) in validity_values_and_mask()
        ) {
            let hybrid_rle = HybridRleDecoder::new(hybrid_rle, 32 - max_idx.leading_zeros(), validity.set_bits());
            let dict = bytemuck::cast_slice(dict.as_slice());
            _test_decode(validity, hybrid_rle, dict, mask)?
        }
    }
}
