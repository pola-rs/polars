use arrow::types::AlignedBytes;

use super::{oob_dict_idx, required_skip_whole_chunks, verify_dict_indices, IndexMapping};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;

/// Decoding kernel for required dictionary encoded.
#[inline(never)]
pub fn decode<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    target: &mut Vec<B>,
    mut num_rows_to_skip: usize,
) -> ParquetResult<()> {
    debug_assert!(num_rows_to_skip <= values.len());

    let num_rows = values.len() - num_rows_to_skip;
    let end_length = target.len() + num_rows;

    if num_rows == 0 {
        return Ok(());
    }

    target.reserve(num_rows);

    if dict.is_empty() {
        return Err(oob_dict_idx());
    }

    // Skip over whole HybridRleChunks
    required_skip_whole_chunks(&mut values, &mut num_rows_to_skip)?;

    while let Some(chunk) = values.next_chunk()? {
        debug_assert!(num_rows_to_skip < chunk.len() || chunk.len() == 0);

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                if size == 0 {
                    continue;
                }

                let Some(value) = dict.get(value) else {
                    return Err(oob_dict_idx());
                };

                target.resize(target.len() + size - num_rows_to_skip, value);
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                if num_rows_to_skip > 0 {
                    decoder.skip_chunks(num_rows_to_skip / 32);
                    num_rows_to_skip %= 32;

                    if let Some((chunk, chunk_size)) = decoder.chunked().next_inexact() {
                        let chunk = &chunk[num_rows_to_skip..chunk_size];
                        verify_dict_indices(chunk, dict.len())?;
                        target.extend(chunk.iter().map(|&idx| {
                            // SAFETY: The dict indices were verified before.
                            unsafe { dict.get_unchecked(idx) }
                        }));
                    }
                }

                let mut chunked = decoder.chunked();
                for chunk in chunked.by_ref() {
                    verify_dict_indices(&chunk, dict.len())?;
                    target.extend(chunk.iter().map(|&idx| {
                        // SAFETY: The dict indices were verified before.
                        unsafe { dict.get_unchecked(idx) }
                    }));
                }

                if let Some((chunk, chunk_size)) = chunked.remainder() {
                    verify_dict_indices(&chunk[..chunk_size], dict.len())?;
                    target.extend(chunk[..chunk_size].iter().map(|&idx| {
                        // SAFETY: The dict indices were verified before.
                        unsafe { dict.get_unchecked(idx) }
                    }));
                }
            },
        }

        num_rows_to_skip = 0;
    }

    debug_assert_eq!(target.len(), end_length);

    Ok(())
}
