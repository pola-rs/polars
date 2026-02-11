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
                HybridRleChunk::Rle(value, length) => {
                    if length == 0 {
                        continue;
                    }

                    // If we know that we have `length` times `value` that we can append, but there
                    // might be nulls in between those values.
                    //
                    // 1. See how many `num_rows = valid + invalid` values `length` would entail.
                    //    This is done with `nth_set_bit_idx` on the validity mask.
                    // 2. Fill `num_rows` values into the target buffer.
                    // 3. Advance the validity mask by `num_rows` values.

                    let Some(value) = dict.get(value) else {
                        return Err(oob_dict_idx());
                    };

                    // We have `length` values but they may span more rows due to interspersed nulls.
                    //
                    // Example: validity = [1,0,0,0,1,0,0,0,1,0,0, 1, 1, 1] and length = 3
                    //          positions:  0 1 2 3 4 5 6 7 8 9 10 11 12 13
                    //          indices:    0       1       2      3  4  5  (of set bits)
                    //
                    // First RLE chunk owns 3 values at positions 0, 4, 8 (sparse, many nulls).
                    // Second RLE chunk owns values at positions 11, 12, 13.
                    //
                    // Correct: nth_set_bit_idx(length-1 = 2, 0) = 8  → rows = 8+1 = 9 (rows 0-8) ✓
                    // Bug:     nth_set_bit_idx(length = 3, 0)   = 11 → rows = 11   (rows 0-10!)
                    //
                    // The bug claims rows 9, 10 which are nulls belonging to the second chunk.
                    let num_chunk_rows = validity
                        .nth_set_bit_idx(length - 1, 0)
                        .map_or(validity.len(), |v| v + 1);

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

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use arrow::types::Bytes4Alignment4;

    use super::decode;
    use crate::parquet::encoding::hybrid_rle::{Encoder, HybridRleDecoder};

    /// Position:  0  1  2  3  4  5  6  7  8  9 10 11 12 13
    /// Validity:  1  0  0  0  1  0  0  0  1  0  0  1  1  1
    /// Value:    66          66          66       99 99 99
    ///           |----chunk0 (3 values)----|      |-chunk1-|
    ///
    /// Chunk0 owns rows 0-8 (9 rows), chunk1 owns rows 9-13 (5 rows).
    /// Positions 9,10 are nulls that belong to chunk1, so they get filled with 99.
    ///
    /// BUG: `nth_set_bit_idx(length=3, 0)` returns 11, so chunk0 claims rows 0-10.
    ///      Positions 9,10 get filled with 66 (wrong - they belong to chunk1).
    ///
    /// FIX: `nth_set_bit_idx(length-1=2, 0) + 1` returns 9, so chunk0 claims rows 0-8.
    ///      Positions 9,10 get filled with 99 (correct - they belong to chunk1).
    #[test]
    fn test_rle_decode_with_sparse_nulls() {
        // Bitmap bits (LSB first within each byte):
        // Byte 0: positions 0-7  = 1,0,0,0,1,0,0,0 = 0b00010001
        // Byte 1: positions 8-13 = 1,0,0,1,1,1     = 0b00111001
        let validity_bytes: Vec<u8> = vec![0b00010001, 0b00111001];
        let validity = Bitmap::try_new(validity_bytes, 14).unwrap();

        let mut encoded = Vec::new();
        u32::run_length_encode(&mut encoded, 3, 0, 1).unwrap(); // 3x dict index 0
        u32::run_length_encode(&mut encoded, 3, 1, 1).unwrap(); // 3x dict index 1

        let dict: &[Bytes4Alignment4] = bytemuck::cast_slice(&[66u32, 99u32]);
        let decoder = HybridRleDecoder::new(&encoded, 1, 6); // 6 total values

        let mut target = Vec::new();
        decode(decoder, dict, validity, &mut target, 0).unwrap();

        assert_eq!(target.len(), 14, "should have 14 rows");

        // Valid positions in chunk0 (dict[0]=66): positions 0, 4, 8
        assert_eq!(target[0], dict[0], "position 0 should be dict[0]");
        assert_eq!(target[4], dict[0], "position 4 should be dict[0]");
        assert_eq!(target[8], dict[0], "position 8 should be dict[0]");

        // Valid positions in chunk1 (dict[1]=99): positions 11, 12, 13
        assert_eq!(target[11], dict[1], "position 11 should be dict[1]");
        assert_eq!(target[12], dict[1], "position 12 should be dict[1]");
        assert_eq!(target[13], dict[1], "position 13 should be dict[1]");

        // Null positions 9,10 belong to chunk1, so they should be filled with dict[1].
        // BUG: These would be dict[0] if chunk0 incorrectly claims rows 0-10.
        assert_eq!(target[9], dict[1], "position 9 (null) should be dict[1]");
        assert_eq!(target[10], dict[1], "position 10 (null) should be dict[1]");
    }
}
