mod basic;
mod nested;

use arrow::array::{Array, FixedSizeBinaryArray, Float16Array, PrimitiveArray};
use arrow::types::{NativeType, i256};
pub use basic::array_to_page;
pub use nested::array_to_page as nested_array_to_page;
use polars_compute::min_max::MinMaxKernel;

use super::{EncodeNullability, StatisticsOptions};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::FixedLenStatistics;

pub(crate) fn encode_plain(
    array: &FixedSizeBinaryArray,
    options: EncodeNullability,
    buffer: &mut Vec<u8>,
) {
    // append the non-null values
    if options.is_optional() && array.validity().is_some() {
        array.iter().for_each(|x| {
            if let Some(x) = x {
                buffer.extend_from_slice(x);
            }
        })
    } else {
        buffer.extend_from_slice(array.values());
    }
}

pub(super) fn build_statistics(
    array: &FixedSizeBinaryArray,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                #[expect(clippy::filter_map_identity)]
                array.iter().filter_map(|x| x).max().map(|x| x.to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                #[expect(clippy::filter_map_identity)]
                array.iter().filter_map(|x| x).min().map(|x| x.to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_float16(
    array: &Float16Array,
    primitive_type: PrimitiveType,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .max_propagate_nan_kernel()
                    .map(|x| x.to_le_bytes().as_ref().to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .min_propagate_nan_kernel()
                    .map(|x| x.to_le_bytes().as_ref().to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_decimal<T>(
    array: &PrimitiveArray<T>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics
where
    T: NativeType + Ord,
{
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.to_be_bytes().as_ref()[16 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.to_be_bytes().as_ref()[16 - size..].to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_decimal256_with_i128(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.0.low().to_be_bytes()[16 - size..].to_vec())
            })
            .flatten(),
    }
}

pub(super) fn build_statistics_decimal256(
    array: &PrimitiveArray<i256>,
    primitive_type: PrimitiveType,
    size: usize,
    options: &StatisticsOptions,
) -> FixedLenStatistics {
    FixedLenStatistics {
        primitive_type,
        null_count: options.null_count.then_some(array.null_count() as i64),
        distinct_count: None,
        max_value: options
            .max_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .max()
                    .map(|x| x.0.to_be_bytes()[32 - size..].to_vec())
            })
            .flatten(),
        min_value: options
            .min_value
            .then(|| {
                array
                    .iter()
                    .flatten()
                    .min()
                    .map(|x| x.0.to_be_bytes()[32 - size..].to_vec())
            })
            .flatten(),
    }
}
