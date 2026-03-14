use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::*;

use super::{Version, WriteOptions};
use crate::parquet::CowBuffer;
use crate::parquet::compression::CompressionOptions;
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::hybrid_rle::{self, encode};
use crate::parquet::metadata::Descriptor;
use crate::parquet::page::{DataPage, DataPageHeader, DataPageHeaderV1, DataPageHeaderV2};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::ParquetStatistics;

/// writes the def levels to a `Vec<u8>` and returns it.
pub fn write_def_levels(
    writer: &mut Vec<u8>,
    is_optional: bool,
    validity: Option<&Bitmap>,
    len: usize,
    version: Version,
) -> PolarsResult<()> {
    if is_optional {
        match version {
            Version::V1 => {
                writer.extend(&[0, 0, 0, 0]);
                let start = writer.len();

                match validity {
                    None => <bool as hybrid_rle::Encoder<bool>>::run_length_encode(
                        writer, len, true, 1,
                    )?,
                    Some(validity) => encode::<bool, _, _>(writer, validity.iter(), 1)?,
                }

                // write the first 4 bytes as length
                let length = ((writer.len() - start) as i32).to_le_bytes();
                (0..4).for_each(|i| writer[start - 4 + i] = length[i]);
            },
            Version::V2 => match validity {
                None => {
                    <bool as hybrid_rle::Encoder<bool>>::run_length_encode(writer, len, true, 1)?
                },
                Some(validity) => encode::<bool, _, _>(writer, validity.iter(), 1)?,
            },
        }

        Ok(())
    } else {
        // is required => no def levels
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_plain_page(
    buffer: Vec<u8>,
    num_values: usize,
    num_rows: usize,
    null_count: usize,
    repetition_levels_byte_length: usize,
    definition_levels_byte_length: usize,
    statistics: Option<ParquetStatistics>,
    type_: PrimitiveType,
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<DataPage> {
    let header = match options.version {
        Version::V1 => DataPageHeader::V1(DataPageHeaderV1 {
            num_values: num_values as i32,
            encoding: encoding.into(),
            definition_level_encoding: Encoding::Rle.into(),
            repetition_level_encoding: Encoding::Rle.into(),
            statistics,
        }),
        Version::V2 => DataPageHeader::V2(DataPageHeaderV2 {
            num_values: num_values as i32,
            encoding: encoding.into(),
            num_nulls: null_count as i32,
            num_rows: num_rows as i32,
            definition_levels_byte_length: definition_levels_byte_length as i32,
            repetition_levels_byte_length: repetition_levels_byte_length as i32,
            is_compressed: Some(options.compression != CompressionOptions::Uncompressed),
            statistics,
        }),
    };
    Ok(DataPage::new(
        header,
        CowBuffer::Owned(buffer),
        Descriptor {
            primitive_type: type_,
            max_def_level: 0,
            max_rep_level: 0,
        },
        num_rows,
    ))
}

/// Auxiliary iterator adapter to declare the size hint of an iterator.
pub(super) struct ExactSizedIter<T, I: Iterator<Item = T>> {
    iter: I,
    remaining: usize,
}

impl<T, I: Iterator<Item = T> + Clone> Clone for ExactSizedIter<T, I> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            remaining: self.remaining,
        }
    }
}

impl<T, I: Iterator<Item = T>> ExactSizedIter<T, I> {
    pub fn new(iter: I, length: usize) -> Self {
        Self {
            iter,
            remaining: length,
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for ExactSizedIter<T, I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().inspect(|_| self.remaining -= 1)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, I: Iterator<Item = T>> std::iter::ExactSizeIterator for ExactSizedIter<T, I> {}

/// Returns the number of bits needed to bitpack `max`
#[inline]
pub fn get_bit_width(max: u64) -> u32 {
    64 - max.leading_zeros()
}

/// Returns `true` if the parquet `PrimitiveType` represents a UTF-8 string
/// column (as opposed to raw binary). This is determined by checking the
/// logical type (`String`) or converted type (`Utf8`).
pub(super) fn is_utf8_type(primitive_type: &PrimitiveType) -> bool {
    use crate::parquet::schema::types::{PrimitiveConvertedType, PrimitiveLogicalType};

    matches!(
        primitive_type.logical_type,
        Some(PrimitiveLogicalType::String)
    ) || matches!(
        primitive_type.converted_type,
        Some(PrimitiveConvertedType::Utf8)
    )
}

pub(super) fn invalid_encoding(encoding: Encoding, dtype: &ArrowDataType) -> PolarsError {
    polars_err!(InvalidOperation:
        "Datatype {:?} cannot be encoded by {:?} encoding",
        dtype,
        encoding
    )
}

/// Extracts the first `len` or fewer bytes of the input that are valid UTF-8.
///
/// Truncation to fewer than `len` bytes occurs when the truncation would land in the middle of a
/// multibyte UTF-8 codepoint. In these cases, truncation is applied to the end of the last valid
/// codepoint. If no valid UTF-8 characters are found within the specified length, returns `None`.
fn extract_truncated_utf8(bytes: &[u8], len: usize) -> Option<&str> {
    // UTF-8: truncate at a character boundary. The first `utf8_chunk`
    // of the byte prefix gives us the longest valid UTF-8 prefix (any
    // trailing incomplete char is in `.invalid()`).
    bytes[..len].utf8_chunks().next().map(|span| span.valid())
}

/// Truncates a min statistics value to `len` bytes.
///
/// When `is_utf8` is true, truncation happens at a character boundary so
/// the result stays valid UTF-8. For binary data, raw byte truncation is
/// used. In both cases a prefix is always <= the original in lexicographic
/// order, so the truncated value remains a valid lower bound.
pub(super) fn truncate_min_statistics_value(mut val: Vec<u8>, len: u64, is_utf8: bool) -> Vec<u8> {
    if val.len() <= len as usize {
        return val;
    }
    if is_utf8 {
        let utf8_substring = extract_truncated_utf8(&val, len as usize);
        if let Some(substring) = utf8_substring {
            val.truncate(substring.len());
        }
    } else {
        // Binary data (or zero-length valid prefix): truncate raw bytes.
        val.truncate(len as usize);
    }
    val
}

/// Increment the last UTF-8 character in `data` without changing its byte
/// length. Characters whose successor would need more bytes (e.g. U+007F →
/// U+0080 grows from 1 to 2 bytes) are skipped, and the preceding character
/// is tried instead. Returns the (potentially shorter) sub-slice with the
/// incremented character, or `None` if no character can be incremented.
fn increment_utf8(data: &mut str) -> Option<&str> {
    for (idx, ch) in data.char_indices().rev() {
        let original_len = ch.len_utf8();
        if let Some(next_char) = char::from_u32(ch as u32 + 1) {
            if next_char.len_utf8() == original_len {
                // SAFETY: `next_char` has the same UTF-8 byte length as `ch`,
                // so writing it into the same position preserves valid UTF-8.
                let bytes = unsafe { data.as_bytes_mut() };
                next_char.encode_utf8(&mut bytes[idx..]);
                return Some(&data[..idx + original_len]);
            }
        }
    }
    None
}

/// Increment the last non-`0xFF` byte in `data` and return the sub-slice up
/// to and including that byte. If every byte is `0xFF` returns `None`.
fn increment_bytes(data: &mut [u8]) -> Option<&[u8]> {
    for idx in (0..data.len()).rev() {
        if data[idx] < 0xFF {
            data[idx] += 1;
            return Some(&data[..=idx]);
        }
    }
    None
}

/// Truncates a max statistics value to `len` bytes, then increments it so
/// that the result is still a valid upper bound.
///
/// When `is_utf8` is true, truncation happens at a character boundary and
/// the last *character* (not byte) is incremented, keeping the result valid
/// UTF-8. For binary data the last non-0xFF byte is incremented.
///
/// Falls back to the original (untruncated) value when no short upper bound
/// can be produced.
pub(super) fn truncate_max_statistics_value(mut val: Vec<u8>, len: u64, is_utf8: bool) -> Vec<u8> {
    if val.len() <= len as usize {
        return val;
    }
    if is_utf8 {
        let valid_len = extract_truncated_utf8(&val, len as usize).map(|s| s.len());
        if let Some(valid_len) = valid_len {
            // SAFETY: `extract_truncated_utf8` guarantees `val[..valid_len]` is valid UTF-8.
            let mutable_str = unsafe { std::str::from_utf8_unchecked_mut(&mut val[..valid_len]) };
            if let Some(incremented_len) = increment_utf8(mutable_str).map(|s| s.len()) {
                val.truncate(incremented_len);
            }
        }
    } else {
        // Binary data: truncate and increment raw bytes.
        if let Some(new_len) = increment_bytes(&mut val[..len as usize]).map(|s| s.len()) {
            val.truncate(new_len);
        }
    }
    // All bytes in the prefix were 0xFF — fall back to original.
    val
}
