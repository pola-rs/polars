use arrow::array::PrimitiveArray;
use arrow::compute::arity::try_unary;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use arrow::error::{Error as ArrowError, Result};
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime, TimeZone};
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
use polars_error::polars_bail;

use crate::error::PolarsResult;
use crate::prelude::ArrayRef;

#[cfg(feature = "timezones")]
fn convert_to_naive_local(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    use_earliest: Option<bool>,
) -> Result<NaiveDateTime> {
    match from_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.with_timezone(to_tz).naive_local()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match use_earliest {
            Some(true) => Ok(dt_earliest.with_timezone(to_tz).naive_local()),
            Some(false) => Ok(dt_latest.with_timezone(to_tz).naive_local()),
            None => Err(ArrowError::InvalidArgumentError(
                format!("datetime '{}' is ambiguous in time zone '{}'. Please use `use_earliest` to tell how it should be localized.", ndt, from_tz)
            ))
        },
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
fn convert_to_timestamp(
    from_tz: Tz,
    to_tz: Tz,
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
    use_earliest: Option<bool>,
) -> PolarsResult<ArrayRef> {
    match tu {
        TimeUnit::Millisecond => {
            let data = try_unary(
                arr,
                |value| {
                    let ndt = timestamp_ms_to_datetime(value);
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt, use_earliest)?
                        .timestamp_millis())
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
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt, use_earliest)?
                        .timestamp_micros())
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
                    Ok(convert_to_naive_local(&from_tz, &to_tz, ndt, use_earliest)?
                        .timestamp_nanos())
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
    from: &str,
    to: &str,
    use_earliest: Option<bool>,
) -> PolarsResult<ArrayRef> {
    Ok(match from.parse::<chrono_tz::Tz>() {
        Ok(from_tz) => match to.parse::<chrono_tz::Tz>() {
            Ok(to_tz) => convert_to_timestamp(from_tz, to_tz, arr, tu, use_earliest)?,
            Err(_) => polars_bail!(ComputeError: "unable to parse time zone: '{}'", to),
        },
        Err(_) => polars_bail!(ComputeError: "unable to parse time zone: '{}'", from),
    })
}
