use polars_arrow::array::{Array, BinaryViewArray};
use polars_compute::min_max::MinMaxKernel;
use polars_error::PolarsResult;

use crate::parquet::encoding::delta_bitpacked;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BinaryStatistics, ParquetStatistics};
use crate::read::schema::is_nullable;
use crate::write::binary::encode_non_null_values;
use crate::write::utils::{
    invalid_encoding, is_utf8_type, truncate_max_binary_statistics_value,
    truncate_min_binary_statistics_value,
};
use crate::write::{EncodeNullability, Encoding, Page, StatisticsOptions, WriteOptions, utils};

pub(crate) fn encode_plain(
    array: &BinaryViewArray,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    if options.is_optional() && array.validity().is_some() {
        // @NOTE: This capacity might overestimate the amount of bytes since the buffers might
        // still contain data that is not referenced by any value.
        let capacity =
            array.total_bytes_len() + (array.len() - array.null_count()) * size_of::<u32>();
        buffer.reserve(capacity);

        encode_non_null_values(array.non_null_values_iter(), buffer);
        // Append the non-null values.
    } else {
        // @NOTE: This capacity might overestimate the amount of bytes since the buffers might
        // still contain data that is not referenced by any value.
        let capacity = array.total_bytes_len() + array.len() * size_of::<u32>();
        buffer.reserve(capacity);

        encode_non_null_values(array.values_iter(), buffer);
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
        _ => return Err(invalid_encoding(encoding, array.dtype())),
    }

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, type_.clone(), &options.statistics))
    } else {
        None
    };

    let mut page = utils::build_plain_page(
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
    )?;
    page.unencoded_byte_array_data_bytes = Some(array.total_bytes_len() as i64);
    Ok(Page::Data(page))
}

pub(crate) fn build_statistics(
    array: &BinaryViewArray,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> ParquetStatistics {
    let (mut min_value, mut max_value) = if options.min_value && options.max_value {
        let min_max = if array.null_count() == 0 {
            min_max_binary(array.values_iter())
        } else {
            min_max_binary(array.non_null_values_iter())
        };
        match min_max {
            Some((min, max)) => (Some(min.to_vec()), Some(max.to_vec())),
            None => (None, None),
        }
    } else {
        (
            options
                .min_value
                .then(|| array.min_propagate_nan_kernel().map(<[u8]>::to_vec))
                .flatten(),
            options
                .max_value
                .then(|| array.max_propagate_nan_kernel().map(<[u8]>::to_vec))
                .flatten(),
        )
    };

    if let Some(len) = options.binary_statistics_truncate_length_usize() {
        let is_utf8 = is_utf8_type(&primitive_type);
        min_value = min_value.map(|v| truncate_min_binary_statistics_value(v, len, is_utf8));
        max_value = max_value.map(|v| truncate_max_binary_statistics_value(v, len, is_utf8));
    }

    BinaryStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value,
        min_value,
    }
    .serialize()
}

fn min_max_binary<'a>(mut values: impl Iterator<Item = &'a [u8]>) -> Option<(&'a [u8], &'a [u8])> {
    let first = values.next()?;
    let (mut min, mut max) = (first, first);
    while let Some(a) = values.next() {
        let Some(b) = values.next() else {
            min = min.min(a);
            max = max.max(a);
            break;
        };
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        min = min.min(lo);
        max = max.max(hi);
    }
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use polars_arrow::datatypes::Field;
    use proptest::prelude::*;

    use super::*;
    use crate::write::{ParquetType, to_parquet_type};

    fn check_statistics(array: &BinaryViewArray) {
        let ParquetType::PrimitiveType(type_) =
            to_parquet_type(&Field::new("s".into(), array.dtype().clone(), true)).unwrap()
        else {
            unreachable!()
        };
        for truncate in [0, 1, 4, 64] {
            for null_count in [false, true] {
                let options = StatisticsOptions {
                    null_count,
                    binary_statistics_truncate_length: Some(truncate),
                    ..StatisticsOptions::default()
                };
                let both = build_statistics(array, type_.clone(), &options);
                let min = build_statistics(
                    array,
                    type_.clone(),
                    &StatisticsOptions {
                        max_value: false,
                        ..options
                    },
                );
                let max = build_statistics(
                    array,
                    type_.clone(),
                    &StatisticsOptions {
                        min_value: false,
                        ..options
                    },
                );
                assert_eq!(both.min_value, min.min_value);
                assert_eq!(both.max_value, max.max_value);
                assert_eq!(both.null_count, min.null_count);
                assert_eq!(both.null_count, max.null_count);
            }
        }
    }

    #[test]
    fn paired_statistics_edge_cases() {
        let values = [
            Some(""),
            None,
            Some("prefix"),
            Some("prefix-long"),
            Some("é"),
            Some("🙂"),
            Some("prefix"),
            None,
        ];
        let array = BinaryViewArray::from_slice(values);
        for offset in 0..=array.len() {
            for length in 0..=array.len() - offset {
                check_statistics(&array.clone().sliced(offset, length));
            }
        }
        check_statistics(&BinaryViewArray::from_slice([None::<&[u8]>; 9]));
    }

    proptest! {
        #[test]
        fn paired_statistics_match_separate_kernels(
            values in prop::collection::vec(
                prop::option::of(prop::collection::vec(any::<u8>(), 0..96)), 0..128),
        ) {
            check_statistics(&BinaryViewArray::from_slice(values));
        }
    }
}
