use parquet2::schema::types::{PrimitiveLogicalType, TimeUnit as ParquetTimeUnit};
use parquet2::statistics::{PrimitiveStatistics, Statistics as ParquetStatistics};
use parquet2::types::NativeType as ParquetNativeType;

use crate::array::*;
use crate::datatypes::TimeUnit;
use crate::error::Result;
use crate::types::NativeType;

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

pub(super) fn push<P: ParquetNativeType, T: NativeType, F: Fn(P) -> Result<T> + Copy>(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
    map: F,
) -> Result<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<T>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutablePrimitiveArray<T>>()
        .unwrap();
    let from = from.map(|s| s.as_any().downcast_ref::<PrimitiveStatistics<P>>().unwrap());
    min.push(from.and_then(|s| s.min_value.map(map)).transpose()?);
    max.push(from.and_then(|s| s.max_value.map(map)).transpose()?);

    Ok(())
}
