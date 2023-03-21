use arrow::array::PrimitiveArray;
use arrow::compute::arity::try_unary;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use arrow::error::{Error as ArrowError, Result};
use arrow::temporal_conversions::{
    parse_offset, timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime, TimeZone};
use polars_error::polars_bail;

use crate::error::PolarsResult;
use crate::prelude::ArrayRef;

#[cfg(feature = "timezones")]
fn convert_to_naive_local<T1: TimeZone + std::fmt::Debug + std::fmt::Display, T2: TimeZone>(
    from_tz: &T1,
    to_tz: &T2,
    ndt: NaiveDateTime,
) -> Result<NaiveDateTime> {
    match from_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.with_timezone(to_tz).naive_local()),
        LocalResult::Ambiguous(_, _) => Err(ArrowError::InvalidArgumentError(
            format!("datetime '{}' is ambiguous in time zone '{}'. Ambiguous datetimes are not yet supported", ndt, from_tz),
        )),
        LocalResult::None => Err(ArrowError::InvalidArgumentError(
            format!(
                "datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported",
                ndt, from_tz
            )
            ,
        )),
    }
}

#[cfg(feature = "timezones")]
fn convert_to_timestamp<T1: TimeZone + std::fmt::Debug + std::fmt::Display, T2: TimeZone>(
    from_tz: T1,
    to_tz: T2,
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
) -> PolarsResult<ArrayRef> {
    match tu {
        TimeUnit::Millisecond => {
            let data = try_unary(
                arr,
                |value| {
                    let ndt = timestamp_ms_to_datetime(value);
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt)?.timestamp_millis())
                },
                ArrowDataType::Int64,
            )?;
            Ok(Box::new(data))
        }
        TimeUnit::Microsecond => {
            let data = try_unary(
                arr,
                |value| {
                    let ndt = timestamp_us_to_datetime(value);
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt)?.timestamp_micros())
                },
                ArrowDataType::Int64,
            )?;
            Ok(Box::new(data))
        }
        TimeUnit::Nanosecond => {
            let data = try_unary(
                arr,
                |value| {
                    let ndt = timestamp_ns_to_datetime(value);
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt)?.timestamp_nanos())
                },
                ArrowDataType::Int64,
            )?;
            Ok(Box::new(data))
        }
        _ => unreachable!(),
    }
}

#[cfg(feature = "timezones")]
pub fn replace_timezone(
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
    from: String,
    to: String,
) -> PolarsResult<ArrayRef> {
    Ok(match from.parse::<chrono_tz::Tz>() {
        Ok(from_tz) => match to.parse::<chrono_tz::Tz>() {
            Ok(to_tz) => convert_to_timestamp(from_tz, to_tz, arr, tu)?,
            Err(_) => match parse_offset(&to) {
                Ok(to_tz) => convert_to_timestamp(from_tz, to_tz, arr, tu)?,
                Err(_) => polars_bail!(ComputeError: "unable to parse time zone: {}", to),
            },
        },
        Err(_) => match parse_offset(&from) {
            Ok(from_tz) => match to.parse::<chrono_tz::Tz>() {
                Ok(to_tz) => convert_to_timestamp(from_tz, to_tz, arr, tu)?,
                Err(_) => match parse_offset(&to) {
                    Ok(to_tz) => convert_to_timestamp(from_tz, to_tz, arr, tu)?,
                    Err(_) => polars_bail!(ComputeError: "unable to parse time zone: {}", to),
                },
            },
            Err(_) => polars_bail!(ComputeError: "unable to parse time zone: {}", from),
        },
    })
}
