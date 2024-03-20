use arrow::legacy::time_zone::Tz;
use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
    SECONDS_IN_DAY,
};

use crate::Duration;

pub(crate) enum RollUnit {
    Month,
    Quarter,
    Year,
}

#[cfg(feature = "timezones")]
use crate::utils::{try_localize_datetime, unlocalize_datetime};

// roll backward to the first day of the month
pub(crate) fn roll_backward(
    t: i64,
    tz: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    roll_unit: RollUnit,
) -> PolarsResult<i64> {
    let ts = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => unlocalize_datetime(timestamp_to_datetime(t), tz),
        _ => timestamp_to_datetime(t),
    };
    let date = match roll_unit {
        RollUnit::Month => NaiveDate::from_ymd_opt(ts.year(), ts.month(), 1).ok_or_else(|| {
            polars_err!(
                ComputeError: format!("Could not construct date {}-{}-1", ts.year(), ts.month())
            )
        })?,
        RollUnit::Quarter => {
            let quarter = ((ts.month() - 1) / 3) * 3 + 1;
            NaiveDate::from_ymd_opt(ts.year(), quarter, 1).ok_or_else(|| {
                polars_err!(
                    ComputeError: format!("Could not construct date {}-{}-1", ts.year(), ts.month())
                )
            })?
        },
        RollUnit::Year => NaiveDate::from_ymd_opt(ts.year(), 1, 1).ok_or_else(|| {
            polars_err!(
                ComputeError: format!("Could not construct date {}-{}-1", ts.year(), ts.month())
            )
        })?,
    };
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
        Some(tz) => datetime_to_timestamp(
            try_localize_datetime(ndt, tz, Ambiguous::Raise, NonExistent::Raise)?
                .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
        ),
        _ => datetime_to_timestamp(ndt),
    };
    Ok(t)
}

pub(crate) fn roll_month_backward(
    t: i64,
    tz: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
) -> PolarsResult<i64> {
    roll_backward(
        t,
        tz,
        timestamp_to_datetime,
        datetime_to_timestamp,
        RollUnit::Month,
    )
}

pub(crate) fn roll_quarter_backward(
    t: i64,
    tz: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
) -> PolarsResult<i64> {
    roll_backward(
        t,
        tz,
        timestamp_to_datetime,
        datetime_to_timestamp,
        RollUnit::Quarter,
    )
}

pub(crate) fn roll_year_backward(
    t: i64,
    tz: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
) -> PolarsResult<i64> {
    roll_backward(
        t,
        tz,
        timestamp_to_datetime,
        datetime_to_timestamp,
        RollUnit::Year,
    )
}

type TimestampsClosure = (
    fn(i64) -> NaiveDateTime, // timestamp_to_datetime
    fn(NaiveDateTime) -> i64, // datetime_to_timestamp
    fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>, // offset_fn
);
pub(crate) fn get_timestamp_closures(time_unit: TimeUnit) -> TimestampsClosure {
    let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
    let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
    let offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>;
    match time_unit {
        TimeUnit::Nanoseconds => {
            timestamp_to_datetime = timestamp_ns_to_datetime;
            datetime_to_timestamp = datetime_to_timestamp_ns;
            offset_fn = Duration::add_ns;
        },
        TimeUnit::Microseconds => {
            timestamp_to_datetime = timestamp_us_to_datetime;
            datetime_to_timestamp = datetime_to_timestamp_us;
            offset_fn = Duration::add_us;
        },
        TimeUnit::Milliseconds => {
            timestamp_to_datetime = timestamp_ms_to_datetime;
            datetime_to_timestamp = datetime_to_timestamp_ms;
            offset_fn = Duration::add_ms;
        },
    };
    (timestamp_to_datetime, datetime_to_timestamp, offset_fn)
}

pub trait PolarsDateStart {
    fn month_start(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
    fn quarter_start(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
    fn year_start(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsDateStart for DatetimeChunked {
    fn month_start(&self, tz: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, _) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply(|t| {
                roll_month_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp)
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }

    fn quarter_start(&self, tz: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, _) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply(|t| {
                roll_quarter_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp)
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }

    fn year_start(&self, tz: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, _) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply(|t| roll_year_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsDateStart for DateChunked {
    fn month_start(&self, _tz: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(self
            .0
            .try_apply(|t| {
                Ok((roll_month_backward(
                    MSECS_IN_DAY * t as i64,
                    None,
                    timestamp_ms_to_datetime,
                    datetime_to_timestamp_ms,
                )? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }

    fn quarter_start(&self, _tz: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(self
            .0
            .try_apply(|t| {
                Ok((roll_quarter_backward(
                    MSECS_IN_DAY * t as i64,
                    None,
                    timestamp_ms_to_datetime,
                    datetime_to_timestamp_ms,
                )? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }

    fn year_start(&self, _tz: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(self
            .0
            .try_apply(|t| {
                Ok((roll_year_backward(
                    MSECS_IN_DAY * t as i64,
                    None,
                    timestamp_ms_to_datetime,
                    datetime_to_timestamp_ms,
                )? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
