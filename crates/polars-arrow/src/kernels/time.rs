use arrow::array::PrimitiveArray;
use arrow::compute::arity::try_unary;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use arrow::error::{Error as ArrowError, Result};
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
use chrono::{LocalResult, NaiveDateTime, TimeZone};
use chrono_tz::Tz;

use crate::error::PolarsResult;

fn convert_to_naive_local(
    from_tz: &Tz,
    to_tz: &Tz,
    ndt: NaiveDateTime,
    use_earliest: Option<bool>,
) -> Result<NaiveDateTime> {
    let ndt = from_tz.from_utc_datetime(&ndt).naive_local();
    match to_tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match use_earliest {
            Some(true) => Ok(dt_earliest.naive_utc()),
            Some(false) => Ok(dt_latest.naive_utc()),
            None => Err(ArrowError::InvalidArgumentError(
                format!("datetime '{}' is ambiguous in time zone '{}'. Please use `use_earliest` to tell how it should be localized.", ndt, to_tz)
            ))
        },
        LocalResult::None => Err(ArrowError::InvalidArgumentError(
            format!(
                "datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported",
                ndt, to_tz
            )
            ,
        )),
    }
}

pub fn replace_time_zone(
    arr: &PrimitiveArray<i64>,
    tu: TimeUnit,
    from_tz: &Tz,
    to_tz: &Tz,
    use_earliest: Option<bool>,
) -> PolarsResult<PrimitiveArray<i64>> {
    let res = match tu {
        TimeUnit::Millisecond => try_unary(
            arr,
            |value| {
                let ndt = timestamp_ms_to_datetime(value);
                Ok(convert_to_naive_local(from_tz, to_tz, ndt, use_earliest)?.timestamp_millis())
            },
            ArrowDataType::Int64,
        ),
        TimeUnit::Microsecond => try_unary(
            arr,
            |value| {
                let ndt = timestamp_us_to_datetime(value);
                Ok(convert_to_naive_local(from_tz, to_tz, ndt, use_earliest)?.timestamp_micros())
            },
            ArrowDataType::Int64,
        ),
        TimeUnit::Nanosecond => try_unary(
            arr,
            |value| {
                let ndt = timestamp_ns_to_datetime(value);
                Ok(convert_to_naive_local(from_tz, to_tz, ndt, use_earliest)?.timestamp_nanos())
            },
            ArrowDataType::Int64,
        ),
        _ => unreachable!(),
    };
    Ok(res?)
}
