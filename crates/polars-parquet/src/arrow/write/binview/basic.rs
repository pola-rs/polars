use arrow::array::{Array, BinaryViewArray};
use polars_compute::min_max::MinMaxKernel;
use polars_error::PolarsResult;

use crate::parquet::encoding::delta_bitpacked;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BinaryStatistics, ParquetStatistics};
use crate::read::schema::is_nullable;
use crate::write::binary::encode_non_null_values;
use crate::write::utils::invalid_encoding;
use crate::write::{utils, EncodeNullability, Encoding, Page, StatisticsOptions, WriteOptions};

pub(crate) fn encode_plain(
    array: &BinaryViewArray,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    if options.is_optional() && array.validity().is_some() {
        let capacity = array.total_bytes_len()
            + (array.len() - array.null_count()) * std::mem::size_of::<u32>();

        let len_before = buffer.len();
        buffer.reserve(capacity);

        encode_non_null_values(array.non_null_values_iter(), buffer);
        // Append the non-null values.
        debug_assert_eq!(buffer.len() - len_before, capacity);
    } else {
        let capacity = array.total_bytes_len() + array.len() * std::mem::size_of::<u32>();

        let len_before = buffer.len();
        buffer.reserve(capacity);

        encode_non_null_values(array.values_iter(), buffer);
        // Append the non-null values.
        debug_assert_eq!(buffer.len() - len_before, capacity);
    }
}

pub(crate) fn encode_delta(
    array: &BinaryViewArray,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    if options.is_optional() && array.validity().is_some() {
        let lengths = utils::ExactSizedIter::new(
            array.non_null_views_iter().map(|v| v.length as i64),
            array.len() - array.null_count(),
        );
        delta_bitpacked::encode(lengths, buffer, 1);

        for slice in array.non_null_values_iter() {
            buffer.extend_from_slice(slice)
        }
    } else {
        let lengths =
            utils::ExactSizedIter::new(array.views().iter().map(|v| v.length as i64), array.len());
        delta_bitpacked::encode(lengths, buffer, 1);

        buffer.extend(array.values_iter().flatten());
    }
}

pub fn array_to_page(
    array: &BinaryViewArray,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> PolarsResult<Page> {
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

    let mut buffer = vec![];
    // TODO! reserve capacity
    utils::write_def_levels(
        &mut buffer,
        is_optional,
        array.validity(),
        array.len(),
        options.version,
    )?;

    let definition_levels_byte_length = buffer.len();

    match encoding {
        Encoding::Plain => encode_plain(array, encode_options, &mut buffer),
        Encoding::DeltaLengthByteArray => encode_delta(array, encode_options, &mut buffer),
        _ => return Err(invalid_encoding(encoding, array.data_type())),
    }

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, type_.clone(), &options.statistics))
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
    .map(Page::Data)
}

pub(crate) fn build_statistics(
    array: &BinaryViewArray,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> ParquetStatistics {
    BinaryStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| array.max_propagate_nan_kernel().map(<[u8]>::to_vec))
            .flatten(),
        min_value: options
            .min_value
            .then(|| array.min_propagate_nan_kernel().map(<[u8]>::to_vec))
            .flatten(),
    }
    .serialize()
}
