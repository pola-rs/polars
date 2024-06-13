use arrow::array::*;
use polars_error::PolarsResult;

use super::super::{utils, WriteOptions};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::encoding::hybrid_rle::bitpacked_encode;
use crate::parquet::encoding::Encoding;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BooleanStatistics, ParquetStatistics};
use crate::write::StatisticsOptions;

fn encode(iterator: impl Iterator<Item = bool>, buffer: &mut Vec<u8>) -> PolarsResult<()> {
    // encode values using bitpacking
    let len = buffer.len();
    let mut buffer = std::io::Cursor::new(buffer);
    buffer.set_position(len as u64);
    Ok(bitpacked_encode(&mut buffer, iterator)?)
}

pub(super) fn encode_plain(
    array: &BooleanArray,
    is_optional: bool,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    if is_optional {
        let iter = array.non_null_values_iter().take(
            array
                .validity()
                .as_ref()
                .map(|x| x.len() - x.unset_bits())
                .unwrap_or_else(|| array.len()),
        );
        encode(iter, buffer)
    } else {
        let iter = array.values().iter();
        encode(iter, buffer)
    }
}

pub fn array_to_page(
    array: &BooleanArray,
    options: WriteOptions,
    type_: PrimitiveType,
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

    encode_plain(array, is_optional, &mut buffer)?;

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, &options.statistics))
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::Plain,
    )
}

pub(super) fn build_statistics(
    array: &BooleanArray,
    options: &StatisticsOptions,
) -> ParquetStatistics {
    use polars_compute::distinct_count::DistinctCountKernel;
    use polars_compute::min_max::MinMaxKernel;

    BooleanStatistics {
        null_count: options.null_count.then(|| array.null_count() as i64),
        distinct_count: options
            .distinct_count
            .then(|| array.distinct_non_null_count().try_into().ok())
            .flatten(),
        max_value: options
            .max_value
            .then(|| array.max_propagate_nan_kernel())
            .flatten(),
        min_value: options
            .min_value
            .then(|| array.min_propagate_nan_kernel())
            .flatten(),
    }
    .serialize()
}
