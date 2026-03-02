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

/// Truncates a min statistics value to `len` bytes.
///
/// When `is_utf8` is true, truncation happens at a character boundary so
/// the result stays valid UTF-8. For binary data, raw byte truncation is
/// used. In both cases a prefix is always <= the original in lexicographic
/// order, so the truncated value remains a valid lower bound.
pub(super) fn truncate_min_statistics_value(val: Vec<u8>, len: usize, is_utf8: bool) -> Vec<u8> {
    if val.len() <= len {
        return val;
    }
    if is_utf8 {
        // UTF-8: truncate at a character boundary. The first `utf8_chunk`
        // of the byte prefix gives us the longest valid UTF-8 prefix (any
        // trailing incomplete char is in `.invalid()`).
        let valid = val[..len].utf8_chunks().next().unwrap().valid();
        if !valid.is_empty() {
            return valid.as_bytes().to_vec();
        }
    }
    // Binary data (or zero-length valid prefix): truncate raw bytes.
    val[..len].to_vec()
}

/// Increment the last UTF-8 character in `data` without changing its byte
/// length. Characters whose successor would need more bytes (e.g. U+007F →
/// U+0080 grows from 1 to 2 bytes) are skipped, and the preceding character
/// is tried instead. Returns `None` if no character can be incremented.
fn increment_utf8(data: &str) -> Option<Vec<u8>> {
    for (idx, ch) in data.char_indices().rev() {
        let original_len = ch.len_utf8();
        if let Some(next_char) = char::from_u32(ch as u32 + 1) {
            if next_char.len_utf8() == original_len {
                let mut result = data.as_bytes()[..idx + original_len].to_vec();
                next_char.encode_utf8(&mut result[idx..]);
                return Some(result);
            }
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
pub(super) fn truncate_max_statistics_value(val: Vec<u8>, len: usize, is_utf8: bool) -> Vec<u8> {
    if val.len() <= len {
        return val;
    }
    if is_utf8 {
        let valid = val[..len].utf8_chunks().next().unwrap().valid();
        if !valid.is_empty() {
            if let Some(incremented) = increment_utf8(valid) {
                return incremented;
            }
        }
        // Cannot produce a short UTF-8 upper bound — fall back to original.
        return val;
    }
    // Binary data: truncate and increment raw bytes.
    let mut truncated = val[..len].to_vec();
    while let Some(&last) = truncated.last() {
        if last == 0xFF {
            truncated.pop();
        } else {
            *truncated.last_mut().unwrap() = last + 1;
            return truncated;
        }
    }
    // All bytes in the prefix were 0xFF — fall back to original.
    val
}
