use arrow::legacy::time_zone::Tz;
use chrono::NaiveDateTime;
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
    SECONDS_IN_DAY,
};

use crate::month_start::roll_backward;
use crate::windows::duration::Duration;

// roll forward to the last day of the month
fn roll_forward(
    t: i64,
    time_zone: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
) -> PolarsResult<i64> {
    let t = roll_backward(t, time_zone, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = offset_fn(&Duration::parse("1mo"), t, time_zone)?;
    offset_fn(&Duration::parse("-1d"), t, time_zone)
}

pub trait PolarsMonthEnd {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthEnd for DatetimeChunked {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self> {
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
        let offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>;
        match self.time_unit() {
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
        Ok(self
            .0
            .try_apply_nonnull_values_generic(|t| {
                roll_forward(
                    t,
                    time_zone,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    offset_fn,
                )
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsMonthEnd for DateChunked {
    fn month_end(&self, _time_zone: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.0.try_apply_nonnull_values_generic(|t| {
            let fwd = roll_forward(
                MSECS_IN_DAY * t as i64,
                None,
                timestamp_ms_to_datetime,
                datetime_to_timestamp_ms,
                Duration::add_ms,
            )?;
            PolarsResult::Ok((fwd / MSECS_IN_DAY) as i32)
        })?;
        Ok(ret.into_date())
    }
}
