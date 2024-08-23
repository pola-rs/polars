use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::*;

use super::{Version, WriteOptions};
use crate::parquet::compression::CompressionOptions;
use crate::parquet::encoding::hybrid_rle::{self, encode};
use crate::parquet::encoding::Encoding;
use crate::parquet::metadata::Descriptor;
use crate::parquet::page::{DataPage, DataPageHeader, DataPageHeaderV1, DataPageHeaderV2};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::ParquetStatistics;
use crate::parquet::CowBuffer;

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

pub(super) fn invalid_encoding(encoding: Encoding, data_type: &ArrowDataType) -> PolarsError {
    polars_err!(InvalidOperation:
        "Datatype {:?} cannot be encoded by {:?} encoding",
        data_type,
        encoding
    )
}
