use arrow::legacy::time_zone::Tz;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
    SECONDS_IN_DAY,
};

#[cfg(feature = "timezones")]
use crate::utils::{localize_datetime, unlocalize_datetime};

// roll backward to the first day of the month
pub(crate) fn roll_backward(
    t: i64,
    tz: Option<&Tz>,
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
    .ok_or_else(|| {
        polars_err!(
            ComputeError:
                format!(
                    "Could not construct time {}:{}:{}.{}",
                    ts.hour(),
                    ts.minute(),
                    ts.second(),
                    ts.timestamp_subsec_nanos()
                )
        )
    })?;
    let ndt = NaiveDateTime::new(date, time);
    let t = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => datetime_to_timestamp(localize_datetime(ndt, tz, "raise")?),
        _ => datetime_to_timestamp(ndt),
    };
    Ok(t)
}

pub trait PolarsMonthStart {
    fn month_start(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthStart for DatetimeChunked {
    fn month_start(&self, tz: Option<&Tz>) -> PolarsResult<Self> {
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
        match self.time_unit() {
            TimeUnit::Nanoseconds => {
                timestamp_to_datetime = timestamp_ns_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_ns;
            },
            TimeUnit::Microseconds => {
                timestamp_to_datetime = timestamp_us_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_us;
            },
            TimeUnit::Milliseconds => {
                timestamp_to_datetime = timestamp_ms_to_datetime;
                datetime_to_timestamp = datetime_to_timestamp_ms;
            },
        };
        Ok(self
            .0
            .try_apply(|t| roll_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsMonthStart for DateChunked {
    fn month_start(&self, _tz: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(self
            .0
            .try_apply(|t| {
                Ok((roll_backward(
                    MSECS_IN_DAY * t as i64,
                    None,
                    timestamp_ms_to_datetime,
                    datetime_to_timestamp_ms,
                )? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
