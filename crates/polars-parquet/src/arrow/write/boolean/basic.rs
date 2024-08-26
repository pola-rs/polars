use arrow::array::*;
use polars_error::{polars_bail, PolarsResult};

use super::super::{utils, WriteOptions};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::encoding::hybrid_rle::{self, bitpacked_encode};
use crate::parquet::encoding::Encoding;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BooleanStatistics, ParquetStatistics};
use crate::write::{EncodeNullability, StatisticsOptions};

fn encode(iterator: impl Iterator<Item = bool>, buffer: &mut Vec<u8>) -> PolarsResult<()> {
    // encode values using bitpacking
    let len = buffer.len();
    let mut buffer = std::io::Cursor::new(buffer);
    buffer.set_position(len as u64);
    Ok(bitpacked_encode(&mut buffer, iterator)?)
}

pub(super) fn encode_plain(
    array: &BooleanArray,
    encode_options: EncodeNullability,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    if encode_options.is_optional() && array.validity().is_some() {
        encode(array.non_null_values_iter(), buffer)
    } else {
        encode(array.values().iter(), buffer)
    }
}

pub(super) fn encode_hybrid_rle(
    array: &BooleanArray,
    encode_options: EncodeNullability,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    buffer.extend_from_slice(&[0; 4]);
    let start = buffer.len();

    if encode_options.is_optional() && array.validity().is_some() {
        hybrid_rle::encode(buffer, array.non_null_values_iter(), 1)?;
    } else {
        hybrid_rle::encode(buffer, array.values().iter(), 1)?;
    }

    let length = buffer.len() - start;

    // write the first 4 bytes as length
    let length = (length as i32).to_le_bytes();
    (0..4).for_each(|i| buffer[start - 4 + i] = length[i]);

    Ok(())
}

pub fn array_to_page(
    array: &BooleanArray,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> PolarsResult<DataPage> {
    let is_optional = is_nullable(&type_.field_info);
    let encode_nullability = EncodeNullability::new(is_optional);

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

    match encoding {
        Encoding::Plain => encode_plain(array, encode_nullability, &mut buffer)?,
        Encoding::Rle => encode_hybrid_rle(array, encode_nullability, &mut buffer)?,
        other => polars_bail!(nyi = "Encoding boolean as {other:?}"),
    }

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
        encoding,
    )
}

pub(super) fn build_statistics(
    array: &BooleanArray,
    options: &StatisticsOptions,
) -> ParquetStatistics {
    use polars_compute::min_max::MinMaxKernel;
    use polars_compute::unique::GenericUniqueKernel;

    BooleanStatistics {
        null_count: options.null_count.then(|| array.null_count() as i64),
        distinct_count: options
            .distinct_count
            .then(|| array.n_unique_non_null().try_into().ok())
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
