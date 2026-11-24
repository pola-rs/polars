use polars_arrow::array::*;
use polars_compute::filter::filter_boolean_kernel;
use polars_error::{PolarsResult, polars_bail};

use super::super::{WriteOptions, utils};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::encoding::{Encoding, hybrid_rle};
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::{BooleanStatistics, ParquetStatistics};
use crate::write::{EncodeNullability, StatisticsOptions};

pub(super) fn encode_plain(
    array: &BooleanArray,
    encode_options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    let compacted;
    let values = if encode_options.is_optional()
        && let Some(validity) = array.validity()
    {
        compacted = filter_boolean_kernel(array.values(), validity);
        &compacted
    } else {
        array.values()
    };

    // encode values using bitpacking
    let (bytes, offset, len) = values.as_slice();
    hybrid_rle::append_packed_bits(buffer, bytes, offset, len);
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
        Encoding::Plain => encode_plain(array, encode_nullability, &mut buffer),
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

#[cfg(test)]
mod tests {
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::ArrowDataType;

    use super::*;

    #[test]
    fn plain_boolean_slices() {
        let values: Bitmap = (0..91).map(|i| i % 5 != 0).collect();
        let masks = [
            None,
            Some(Bitmap::new_with_value(true, 91)),
            Some(Bitmap::new_zeroed(91)),
            Some((0..91).map(|i| i % 3 != 0).collect()),
        ];
        for validity in masks {
            let array = BooleanArray::new(ArrowDataType::Boolean, values.clone(), validity);
            for offset in 0..8 {
                for len in 0..=81 {
                    let slice = array.clone().sliced(offset, len);
                    let mut encoded = vec![0xa5];
                    encode_plain(&slice, EncodeNullability::Optional, &mut encoded);
                    let mut expected = vec![0xa5];
                    let values: Vec<_> = slice.iter().flatten().collect();
                    for group in values.chunks(8) {
                        expected.push(
                            group
                                .iter()
                                .enumerate()
                                .fold(0, |byte, (i, &bit)| byte | (u8::from(bit) << i)),
                        );
                    }
                    assert_eq!(encoded, expected);
                }
            }
        }
    }
}
