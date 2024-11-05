use arrow::types::AlignedBytes;

use super::{oob_dict_idx, verify_dict_indices, verify_dict_indices_slice};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode<B: AlignedBytes>(
    mut values: HybridRleDecoder<'_>,
    dict: &[B],
    target: &mut Vec<B>,
    mut num_skipped_rows: usize,
) -> ParquetResult<()> {
    debug_assert!(num_skipped_rows <= values.len());

    if num_skipped_rows == values.len() {
        return Ok(());
    }
    if dict.is_empty() && values.len() > 0 {
        return Err(oob_dict_idx());
    }

    let start_length = target.len();
    let end_length = start_length + values.len() - num_skipped_rows;

    target.reserve(values.len() - num_skipped_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    while let Some(chunk) = values.next_chunk()? {
        let chunk_len = chunk.len();

        if chunk_len <= num_skipped_rows {
            num_skipped_rows -= chunk_len;
            continue;
        }

        match chunk {
            HybridRleChunk::Rle(value, length) => {
                let target_slice;
                // SAFETY:
                // 1. `target_ptr..target_ptr + values.len()` is allocated
                // 2. `length <= limit`
                unsafe {
                    target_slice =
                        std::slice::from_raw_parts_mut(target_ptr, length - num_skipped_rows);
                    target_ptr = target_ptr.add(length);
                }

                let Some(&value) = dict.get(value as usize) else {
                    return Err(oob_dict_idx());
                };

                target_slice.fill(value);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                if num_skipped_rows > 0 {
                    decoder.skip_chunks(num_skipped_rows / 32);
                    num_skipped_rows %= 32;
                    if let Some((chunk, chunk_size)) = decoder.chunked().next_inexact() {
                        let chunk = &chunk[num_skipped_rows..chunk_size];
                        verify_dict_indices_slice(chunk, dict.len())?;

                        for (i, &idx) in chunk.iter().enumerate() {
                            unsafe { target_ptr.add(i).write(*dict.get_unchecked(idx as usize)) };
                        }
                        unsafe {
                            target_ptr = target_ptr.add(chunk_size);
                        }
                    }
                }

                let mut chunked = decoder.chunked();
                for chunk in chunked.by_ref() {
                    verify_dict_indices(&chunk, dict.len())?;

                    for (i, &idx) in chunk.iter().enumerate() {
                        unsafe { target_ptr.add(i).write(*dict.get_unchecked(idx as usize)) };
                    }
                    unsafe {
                        target_ptr = target_ptr.add(32);
                    }
                }

                if let Some((chunk, chunk_size)) = chunked.remainder() {
                    verify_dict_indices_slice(&chunk[..chunk_size], dict.len())?;

                    for (i, &idx) in chunk[..chunk_size].iter().enumerate() {
                        unsafe { target_ptr.add(i).write(*dict.get_unchecked(idx as usize)) };
                    }
                    unsafe {
                        target_ptr = target_ptr.add(chunk_size);
                    }
                }
            },
        }

        num_skipped_rows = 0;
    }

    unsafe {
        target.set_len(end_length);
    }

    Ok(())
}
