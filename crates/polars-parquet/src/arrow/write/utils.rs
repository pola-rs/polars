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

/// Truncates to the last valid UTF-8 codepoint in `bytes[..requested_len]` if one can be found, or
/// otherwise the smallest `n` for which `bytes[..n]` is valid UTF-8.
///
/// If no truncation is performed, a `None` is returned.
fn truncate_utf8_aware(bytes: &[u8], requested_len: usize) -> Option<&[u8]> {
    if bytes.len() <= requested_len {
        return None;
    }

    if let Some(chunk) = bytes[..requested_len]
        .utf8_chunks()
        .next()
        .map(|span| span.valid().as_bytes())
        .filter(|x| !x.is_empty())
    {
        return Some(chunk);
    }

    bytes[..usize::min(bytes.len(), 4)]
        .utf8_chunks()
        .next()
        .map(|span| span.valid().as_bytes())
        .filter(|x| !x.is_empty() && x.len() < bytes.len())
}

/// Truncates a min statistics value to `len` bytes.
///
/// When `is_utf8` is true, truncation happens at a character boundary so
/// the result stays valid UTF-8. For binary data, raw byte truncation is
/// used. In both cases a prefix is always <= the original in lexicographic
/// order, so the truncated value remains a valid lower bound.
pub(super) fn truncate_min_binary_statistics_value(
    mut val: Vec<u8>,
    len: usize,
    is_utf8: bool,
) -> Vec<u8> {
    if val.len() <= len {
        return val;
    }

    if is_utf8 {
        if let Some(prefix) = truncate_utf8_aware(&val, len) {
            val.truncate(prefix.len());
        }
    } else {
        val.truncate(len);
    }

    val
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
pub(super) fn truncate_max_binary_statistics_value(
    mut val: Vec<u8>,
    len: usize,
    is_utf8: bool,
) -> Vec<u8> {
    if val.len() <= len {
        return val;
    }

    if is_utf8 {
        if let Some(end_idx) = truncate_utf8_aware(&val, len).map(|p| p.len())
            && let Some(end_idx) =
                increment_utf8(std::str::from_utf8_mut(val.get_mut(..end_idx).unwrap()).unwrap())
        {
            val.truncate(end_idx);
        }
    } else if let Some((i, new_c)) = (0..len)
        .rev()
        .chain(len..val.len() - 1)
        .find_map(|i| val[i].checked_add(1).map(|c| (i, c)))
    {
        val[i] = new_c;
        val.truncate(i + 1)
    }

    val
}

/// Find and increment last UTF-8 character that can be incremented without changing the encoded
/// UTF-8 byte length. Returns the byte position of the end of the incremented char.
fn increment_utf8(s: &mut str) -> Option<usize> {
    let (idx, new_char) = s.char_indices().rev().find_map(|(idx, c)| {
        char::from_u32(c as u32 + 1)
            .filter(|new_c| new_c.len_utf8() == c.len_utf8())
            .map(|new_c| (idx, new_c))
    })?;

    let trailing = unsafe { &mut s.as_bytes_mut()[idx..] };
    let new_char_byte_len = new_char.encode_utf8(trailing).len();

    Some(idx + new_char_byte_len)
}
