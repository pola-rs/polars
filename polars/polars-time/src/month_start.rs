use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_arrow::time_zone::{PolarsTimeZone, NO_TIMEZONE};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};

use crate::truncate::PolarsTruncate;
#[cfg(feature = "timezones")]
use crate::utils::{localize_datetime, unlocalize_datetime};
use crate::windows::duration::Duration;

pub(crate) fn roll_backward<T: PolarsTimeZone>(
    t: i64,
    tz: Option<&T>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
) -> PolarsResult<i64> {
    let ts = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => unlocalize_datetime(timestamp_to_datetime(t), tz),
        _ => timestamp_to_datetime(t),
    };
    let date = NaiveDate::from_ymd_opt(ts.year(), ts.month(), 1).ok_or(polars_err!(
        ComputeError: format!("Could not construct date {}-{}-1", ts.year(), ts.month())
    ))?;
    let time = NaiveTime::from_hms_nano_opt(
        ts.hour(),
        ts.minute(),
        ts.second(),
        ts.timestamp_subsec_nanos(),
    )
    .ok_or(polars_err!(
        ComputeError:
            format!(
                "Could not construct time {}:{}:{}.{}",
                ts.hour(),
                ts.minute(),
                ts.second(),
                ts.timestamp_subsec_nanos()
            )
    ))?;
    let ndt = NaiveDateTime::new(date, time);
    let t = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => datetime_to_timestamp(localize_datetime(ndt, tz)?),
        _ => datetime_to_timestamp(ndt),
    };
    Ok(t)
}

pub trait PolarsMonthStart {
    fn month_start(
        &self,
        time_unit: Option<TimeUnit>,
        time_zone: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthStart for DatetimeChunked {
    fn month_start(
        &self,
        time_unit: Option<TimeUnit>,
        tz: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self> {
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
        let time_unit = match time_unit {
            Some(time_unit) => time_unit,
            None => polars_bail!(ComputeError: "Expected `time_unit`, got None"),
        };
        match time_unit {
            TimeUnit::Nanoseconds => {
                timestamp_to_datetime = timestamp_ns_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_ns;
            }
            TimeUnit::Microseconds => {
                timestamp_to_datetime = timestamp_us_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_us;
            }
            TimeUnit::Milliseconds => {
                timestamp_to_datetime = timestamp_ms_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_ms;
            }
        };
        Ok(self
            .0
            .try_apply(|t| roll_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp))?
            .into_datetime(time_unit, tz.map(|x| x.to_string())))
    }
}

impl PolarsMonthStart for DateChunked {
    fn month_start(
        &self,
        _time_unit: Option<TimeUnit>,
        _tz: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self> {
        let no_offset = Duration::parse("0ns");
        PolarsTruncate::truncate(self, Duration::parse("1mo"), no_offset, NO_TIMEZONE)
    }
}
