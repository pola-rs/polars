use arrow::array::PrimitiveArray;
use arrow::compute::arity::unary;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

use crate::prelude::ArrayRef;

pub fn cast_timezone(
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
    from: chrono_tz::Tz,
    to: chrono_tz::Tz,
) -> ArrayRef {
    use chrono::TimeZone;

    match tu {
        TimeUnit::Millisecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_ms_to_datetime(value);
                let tz_aware = from.from_local_datetime(&ndt).unwrap();
                let new_tz_aware = tz_aware.with_timezone(&to);
                new_tz_aware.naive_local().timestamp_millis()
            },
            ArrowDataType::Int64,
        )),
        TimeUnit::Microsecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_us_to_datetime(value);
                let tz_aware = from.from_local_datetime(&ndt).unwrap();
                let new_tz_aware = tz_aware.with_timezone(&to);
                new_tz_aware.naive_local().timestamp_micros()
            },
            ArrowDataType::Int64,
        )),
        TimeUnit::Nanosecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_ns_to_datetime(value);
                let tz_aware = from.from_local_datetime(&ndt).unwrap();
                let new_tz_aware = tz_aware.with_timezone(&to);
                new_tz_aware.naive_local().timestamp_nanos()
            },
            ArrowDataType::Int64,
        )),
        _ => unreachable!(),
    }
}
