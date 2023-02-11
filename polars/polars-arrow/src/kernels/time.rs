use arrow::array::PrimitiveArray;
use arrow::compute::arity::unary;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use arrow::temporal_conversions::{
    parse_offset, timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::{NaiveDateTime, TimeZone};

use crate::error::{PolarsError, Result};
use crate::prelude::ArrayRef;

#[cfg(feature = "timezones")]
fn convert_to_naive_local<T1: TimeZone, T2: TimeZone>(
    from_tz: &T1,
    to_tz: &T2,
    ndt: NaiveDateTime,
) -> NaiveDateTime {
    from_tz
        .from_local_datetime(&ndt)
        .unwrap()
        .with_timezone(to_tz)
        .naive_local()
}

#[cfg(feature = "timezones")]
fn convert_to_timestamp<T1: TimeZone, T2: TimeZone>(
    from_tz: T1,
    to_tz: T2,
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
) -> ArrayRef {
    match tu {
        TimeUnit::Millisecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_ms_to_datetime(value);
                convert_to_naive_local(&from_tz, &to_tz, ndt).timestamp_millis()
            },
            ArrowDataType::Int64,
        )),
        TimeUnit::Microsecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_us_to_datetime(value);
                convert_to_naive_local(&from_tz, &to_tz, ndt).timestamp_micros()
            },
            ArrowDataType::Int64,
        )),
        TimeUnit::Nanosecond => Box::new(unary(
            arr,
            |value| {
                let ndt = timestamp_ns_to_datetime(value);
                convert_to_naive_local(&from_tz, &to_tz, ndt).timestamp_nanos()
            },
            ArrowDataType::Int64,
        )),
        _ => unreachable!(),
    }
}

#[cfg(feature = "timezones")]
pub fn replace_timezone(
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
    from: String,
    to: String,
) -> Result<ArrayRef> {
    match from.parse::<chrono_tz::Tz>() {
        Ok(from_tz) => match to.parse::<chrono_tz::Tz>() {
            Ok(to_tz) => Ok(convert_to_timestamp(from_tz, to_tz, arr, tu)),
            Err(_) => match parse_offset(&to) {
                Ok(to_tz) => Ok(convert_to_timestamp(from_tz, to_tz, arr, tu)),
                Err(_) => Err(PolarsError::ComputeError(
                    format!("Could not parse time zone {to}").into(),
                )),
            },
        },
        Err(_) => match parse_offset(&from) {
            Ok(from_tz) => match to.parse::<chrono_tz::Tz>() {
                Ok(to_tz) => Ok(convert_to_timestamp(from_tz, to_tz, arr, tu)),
                Err(_) => match parse_offset(&to) {
                    Ok(to_tz) => Ok(convert_to_timestamp(from_tz, to_tz, arr, tu)),
                    Err(_) => Err(PolarsError::ComputeError(
                        format!("Could not parse time zone {to}").into(),
                    )),
                },
            },
            Err(_) => Err(PolarsError::ComputeError(
                format!("Could not parse time zone {from}").into(),
            )),
        },
    }
}
