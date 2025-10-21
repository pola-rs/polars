use arrow::array::BinaryArray;

use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;
use crate::read::deserialize::dictionary_encoded::verify_dict_indices;

pub fn decode_dictionary(
    values: HybridRleDecoder<'_>,
    target: &mut Vec<u8>,
    offsets: &mut Vec<i64>,
    dict: &BinaryArray<i64>,
) -> ParquetResult<()> {
    assert!(target.is_empty());
    assert!(offsets.is_empty());

    offsets.reserve(values.len() + 1);
    offsets.push(0);

    let mut offset = 0;
    let dict_offsets = dict.offsets();
    let mut total_buffer_size = 0;
    for chunk in values.clone().into_chunk_iter() {
        match chunk? {
            HybridRleChunk::Rle(item, num_repeats) => {
                let length = dict_offsets.length_at(item as usize);
                total_buffer_size += length * num_repeats;
                let end_offset = offset + (length * num_repeats) as i64;
                offsets.extend((offset + length as i64..=end_offset).step_by(length));
                offset = end_offset;
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let mut chunked = decoder.chunked();
                for chunk in &mut chunked {
                    verify_dict_indices(&chunk, dict_offsets.len())?;
                    offsets.extend(chunk.iter().map(|&item| {
                        let length = unsafe { dict_offsets.length_at_unchecked(item as usize) };
                        total_buffer_size += length;
                        offset += length as i64;
                        offset
                    }));
                }

                if let Some((chunk, size)) = chunked.remainder() {
                    verify_dict_indices(&chunk[..size], dict_offsets.len())?;
                    offsets.extend(chunk[..size].iter().map(|&item| {
                        let length = unsafe { dict_offsets.length_at_unchecked(item as usize) };
                        total_buffer_size += length;
                        offset += length as i64;
                        offset
                    }));
                }
            },
        }
    }

    target.reserve(total_buffer_size);
    for chunk in values.into_chunk_iter() {
        match chunk? {
            HybridRleChunk::Rle(item, num_repeats) => {
                let (start, end) = dict_offsets.start_end(item as usize);
                let item = &dict.values()[start..end];
                for _ in 0..num_repeats {
                    target.extend_from_slice(item);
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let mut chunked = decoder.chunked();
                for chunk in &mut chunked {
                    verify_dict_indices(&chunk, dict_offsets.len())?;
                    for item in chunk {
                        let (start, end) =
                            unsafe { dict_offsets.start_end_unchecked(item as usize) };
                        let item = &dict.values()[start..end];
                        target.extend_from_slice(item);
                    }
                }

                if let Some((chunk, size)) = chunked.remainder() {
                    verify_dict_indices(&chunk[..size], dict_offsets.len())?;
                    for &item in &chunk[..size] {
                        let (start, end) =
                            unsafe { dict_offsets.start_end_unchecked(item as usize) };
                        let item = &dict.values()[start..end];
                        target.extend_from_slice(item);
                    }
                }
            },
        }
    }

    Ok(())
}
