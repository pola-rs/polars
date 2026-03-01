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

pub(super) fn invalid_encoding(encoding: Encoding, dtype: &ArrowDataType) -> PolarsError {
    polars_err!(InvalidOperation:
        "Datatype {:?} cannot be encoded by {:?} encoding",
        dtype,
        encoding
    )
}

/// Truncates a min statistics value to `len` bytes.
/// A prefix is always <= the original in lexicographic order,
/// so the truncated value remains a valid lower bound.
pub(super) fn truncate_min_statistics_value(mut val: Vec<u8>, len: usize) -> Vec<u8> {
    if val.len() > len {
        val.truncate(len);
    }
    val
}

/// Truncates a max statistics value to `len` bytes, incrementing the
/// last non-0xFF byte to maintain a valid upper bound.
/// Falls back to the original (untruncated) value if all bytes in the prefix are 0xFF.
pub(super) fn truncate_max_statistics_value(val: Vec<u8>, len: usize) -> Vec<u8> {
    if val.len() <= len {
        return val;
    }
    let mut truncated = val[..len].to_vec();
    // Increment the last non-0xFF byte so the truncated value is still
    // an upper bound. E.g. [0x01, 0x02] -> [0x01, 0x03].
    // If trailing bytes are 0xFF, roll them over: [0x01, 0xFF] -> [0x02].
    while let Some(&last) = truncated.last() {
        if last == 0xFF {
            truncated.pop();
        } else {
            *truncated.last_mut().unwrap() = last + 1;
            return truncated;
        }
    }
    // All bytes in the prefix were 0xFF — we can't produce a valid short
    // upper bound, so fall back to the original untruncated value.
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_min_value_shorter_than_limit() {
        let val = vec![0x01, 0x02];
        assert_eq!(truncate_min_statistics_value(val, 5), vec![0x01, 0x02]);
    }

    #[test]
    fn test_truncate_min_value_exact_limit() {
        let val = vec![0x01, 0x02, 0x03];
        assert_eq!(
            truncate_min_statistics_value(val, 3),
            vec![0x01, 0x02, 0x03]
        );
    }

    #[test]
    fn test_truncate_min_value_longer_than_limit() {
        let val = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(truncate_min_statistics_value(val, 2), vec![0x01, 0x02]);
    }

    #[test]
    fn test_truncate_max_value_shorter_than_limit() {
        let val = vec![0x01, 0x02];
        assert_eq!(truncate_max_statistics_value(val, 5), vec![0x01, 0x02]);
    }

    #[test]
    fn test_truncate_max_value_exact_limit() {
        let val = vec![0x01, 0x02, 0x03];
        assert_eq!(
            truncate_max_statistics_value(val, 3),
            vec![0x01, 0x02, 0x03]
        );
    }

    #[test]
    fn test_truncate_max_value_longer_than_limit() {
        let val = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        // prefix [0x01, 0x02] -> increment last byte -> [0x01, 0x03]
        assert_eq!(truncate_max_statistics_value(val, 2), vec![0x01, 0x03]);
    }

    #[test]
    fn test_truncate_max_value_trailing_0xff_rollover() {
        let val = vec![0x01, 0xFF, 0x03];
        // prefix [0x01, 0xFF] -> 0xFF rolls over, pop -> [0x01] -> increment -> [0x02]
        assert_eq!(truncate_max_statistics_value(val, 2), vec![0x02]);
    }

    #[test]
    fn test_truncate_max_value_all_0xff_fallback() {
        let val = vec![0xFF, 0xFF, 0x03];
        // prefix [0xFF, 0xFF] -> all 0xFF, can't increment -> fallback to original
        assert_eq!(
            truncate_max_statistics_value(val, 2),
            vec![0xFF, 0xFF, 0x03]
        );
    }
}
