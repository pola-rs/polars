use arrow::array::{Array, FixedSizeBinaryArray, PrimitiveArray};
use arrow::types::i256;
use parquet2::encoding::Encoding;
use parquet2::page::DataPage;
use parquet2::schema::types::PrimitiveType;
use parquet2::statistics::{serialize_statistics, FixedLenStatistics};
use polars_error::PolarsResult;

use super::binary::ord_binary;
use super::{utils, WriteOptions};
use crate::arrow::read::schema::is_nullable;

pub(crate) fn encode_plain(array: &FixedSizeBinaryArray, is_optional: bool, buffer: &mut Vec<u8>) {
    // append the non-null values
    if is_optional {
        array.iter().for_each(|x| {
            if let Some(x) = x {
                buffer.extend_from_slice(x);
            }
        })
    } else {
        buffer.extend_from_slice(array.values());
    }
}

pub fn array_to_page(
    array: &FixedSizeBinaryArray,
    options: WriteOptions,
    type_: PrimitiveType,
    statistics: Option<FixedLenStatistics>,
) -> PolarsResult<DataPage> {
    let is_optional = is_nullable(&type_.field_info);
    let validity = array.validity();

    let mut buffer = vec![];
    utils::write_def_levels(
        &mut buffer,
        is_optional,
        validity,
        array.len(),
        options.version,
    )?;

    let definition_levels_byte_length = buffer.len();

    encode_plain(array, is_optional, &mut buffer);

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics.map(|x| serialize_statistics(&x)),
        type_,
        options,
        Encoding::Plain,
    )
}

pub(super) fn build_statistics(
    array: &FixedSizeBinaryArray,
    primitive_type: PrimitiveType,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .max_by(|x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
        min_value: array
            .iter()
            .flatten()
            .min_by(|x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
    }
}

pub(super) fn build_statistics_decimal(
    array: &PrimitiveArray<i128>,
    primitive_type: PrimitiveType,
    size: usize,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .max()
            .map(|x| x.to_be_bytes()[16 - size..].to_vec()),
        min_value: array
            .iter()
            .flatten()
            .min()
            .map(|x| x.to_be_bytes()[16 - size..].to_vec()),
    }
}

pub(super) fn build_statistics_decimal256_with_i128(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .max()
            .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec()),
        min_value: array
            .iter()
            .flatten()
            .min()
            .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec()),
    }
}

pub(super) fn build_statistics_decimal256(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .max()
            .map(|x| x.0.to_be_bytes()[32 - size..].to_vec()),
        min_value: array
            .iter()
            .flatten()
            .min()
            .map(|x| x.0.to_be_bytes()[32 - size..].to_vec()),
    }
}
