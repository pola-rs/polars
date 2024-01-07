use arrow::array::{Array, BinaryViewArray, BinaryViewArrayGeneric, ViewType};
use arrow::compute::aggregate::estimated_bytes_size;
use crate::parquet::schema::types::PrimitiveType;
use polars_error::PolarsResult;
use crate::parquet::encoding::delta_bitpacked;
use crate::parquet::statistics::{BinaryStatistics, ParquetStatistics, serialize_statistics, Statistics};
use crate::read::schema::is_nullable;
use crate::write::{Encoding, WriteOptions, Page, utils};
use crate::write::binary::{encode_non_null_values, ord_binary};
use crate::write::utils::invalid_encoding;

pub(crate) fn encode_plain(
    array: &BinaryViewArray,
    buffer: &mut Vec<u8>
) {
    let len_before = buffer.len();
    let capacity = estimated_bytes_size(array) - array.validity().map(|b|b.as_slice().0.len()).unwrap_or(0)
    + (array.len() - array.null_count()) * std::mem::size_of::<u32>();
    buffer.reserve(capacity);

    encode_non_null_values(array.non_null_values_iter(), buffer);
    // Append the non-null values.
    debug_assert_eq!(buffer.len() - len_before, capacity);
}

pub(crate) fn encode_delta(
    array: &BinaryViewArray,
    buffer: &mut Vec<u8>
) {
    let len = array.len() - array.null_count();
    let lengths = array.non_null_views_iter().map(|v| (*v as u32) as i64);
    let lengths = utils::ExactSizedIter::new(lengths, len);
    delta_bitpacked::encode(lengths, buffer);
}

pub fn array_to_page(
    array: &BinaryViewArray,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding
) -> PolarsResult<Page> {
    let is_optional = is_nullable(&type_.field_info);

    let mut buffer = vec![];
    // TODO! reserve capacity
    utils::write_def_levels(&mut buffer, is_optional, array.validity(), array.len(), options.version)?;

    let definition_levels_byte_length = buffer.len();

    match encoding {
        Encoding::Plain => encode_plain(array, &mut buffer),
        Encoding::DeltaLengthByteArray => encode_delta(array, &mut buffer),
        _ => {
            return Err(invalid_encoding(encoding, array.data_type()))
        }
    }

    let statistics = if options.write_statistics {
        Some(build_statistics(array, type_.clone()))
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

// TODO! speed this up
pub(crate) fn build_statistics(
    array: &BinaryViewArray,
    primitive_type: PrimitiveType,
) -> ParquetStatistics {
    let statistics = &BinaryStatistics {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .max_by( |x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
        min_value: array
            .iter()
            .flatten()
            .min_by( |x, y| ord_binary(x, y))
            .map(|x| x.to_vec()),
    } as &dyn Statistics;
    serialize_statistics(statistics)
}
