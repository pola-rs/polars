use chrono::NaiveDateTime;
use polars_arrow::time_zone::{PolarsTimeZone, NO_TIMEZONE};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
    SECONDS_IN_DAY,
};

use crate::month_start::roll_backward;
use crate::windows::duration::Duration;

// roll forward to the last day of the month
fn roll_forward<T: PolarsTimeZone>(
    t: i64,
    time_zone: Option<&T>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    adder: fn(&Duration, i64, Option<&T>) -> PolarsResult<i64>,
) -> PolarsResult<i64> {
    let t = roll_backward(t, time_zone, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = adder(&Duration::parse("1mo"), t, time_zone)?;
    adder(&Duration::parse("-1d"), t, time_zone)
}

pub trait PolarsMonthEnd {
    fn month_end(&self, time_zone: Option<&impl PolarsTimeZone>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthEnd for DatetimeChunked {
    fn month_end(&self, time_zone: Option<&impl PolarsTimeZone>) -> PolarsResult<Self> {
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
        match self.time_unit() {
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
        fn adder<T: PolarsTimeZone>(
            time_unit: TimeUnit,
        ) -> fn(&Duration, i64, Option<&T>) -> PolarsResult<i64> {
            match time_unit {
                TimeUnit::Nanoseconds => Duration::add_ns,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Milliseconds => Duration::add_ms,
            }
        }
        let adder = adder(self.time_unit());
        Ok(self
            .0
            .try_apply(|t| {
                roll_forward(
                    t,
                    time_zone,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    adder,
                )
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsMonthEnd for DateChunked {
    fn month_end(&self, _time_zone: Option<&impl PolarsTimeZone>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(self
            .0
            .try_apply(|t| {
                Ok((roll_forward(
                    MSECS_IN_DAY * t as i64,
                    NO_TIMEZONE,
                    timestamp_ms_to_datetime,
                    datetime_to_timestamp_ms,
                    Duration::add_ms,
                )? / MSECS_IN_DAY) as i32)
            })?
            .into_date())
    }
}
