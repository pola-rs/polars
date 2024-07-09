use arrow::array::*;
use arrow::datatypes::TimeUnit;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use crate::parquet::schema::types::{PrimitiveLogicalType, TimeUnit as ParquetTimeUnit};
use crate::parquet::statistics::PrimitiveStatistics;
use crate::parquet::types::NativeType as ParquetNativeType;

pub fn timestamp(logical_type: Option<&PrimitiveLogicalType>, time_unit: TimeUnit, x: i64) -> i64 {
    let unit = if let Some(PrimitiveLogicalType::Timestamp { unit, .. }) = logical_type {
        unit
    } else {
        return x;
    };

    match (unit, time_unit) {
        (ParquetTimeUnit::Milliseconds, TimeUnit::Second) => x / 1_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Second) => x / 1_000_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Second) => x * 1_000_000_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Millisecond) => x,
        (ParquetTimeUnit::Microseconds, TimeUnit::Millisecond) => x / 1_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Millisecond) => x / 1_000_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Microsecond) => x * 1_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Microsecond) => x,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Microsecond) => x / 1_000,

        (ParquetTimeUnit::Milliseconds, TimeUnit::Nanosecond) => x * 1_000_000,
        (ParquetTimeUnit::Microseconds, TimeUnit::Nanosecond) => x * 1_000,
        (ParquetTimeUnit::Nanoseconds, TimeUnit::Nanosecond) => x,
    }
}

pub(super) fn push<P: ParquetNativeType, T: NativeType, F: Fn(P) -> PolarsResult<T> + Copy>(
    from: Option<&PrimitiveStatistics<P>>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
    map: F,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<T>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<T>>()
        .unwrap();

    min.push(from.and_then(|s| s.min_value.map(map)).transpose()?);
    max.push(from.and_then(|s| s.max_value.map(map)).transpose()?);

    Ok(())
}
