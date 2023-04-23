use chrono::NaiveDateTime;
use polars_arrow::time_zone::{PolarsTimeZone, NO_TIMEZONE};
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime, MILLISECONDS,
    SECONDS_IN_DAY,
};

use crate::month_start::roll_backward;
use crate::truncate::PolarsTruncate;
use crate::windows::duration::Duration;

fn roll_forward<T: PolarsTimeZone>(
    t: i64,
    tz: Option<&T>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    adder: fn(&Duration, i64, Option<&T>) -> PolarsResult<i64>,
    add_one_month: &Duration,
    subtract_one_day: &Duration,
) -> PolarsResult<i64> {
    let t = roll_backward(t, tz, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = adder(add_one_month, t, tz)?;
    adder(subtract_one_day, t, tz)
}

pub trait PolarsMonthEnd {
    fn month_end(
        &self,
        time_unit: Option<TimeUnit>,
        time_zone: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthEnd for DatetimeChunked {
    fn month_end(
        &self,
        time_unit: Option<TimeUnit>,
        tz: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self> {
        let timestamp_to_datetime: fn(i64) -> NaiveDateTime;
        let datetime_to_timestamp: fn(NaiveDateTime) -> i64;
        let add_one_month = Duration::parse("1mo");
        let subtract_one_day = Duration::parse("-1d");
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
        fn adder<T: PolarsTimeZone>(
            time_unit: TimeUnit,
        ) -> fn(&Duration, i64, Option<&T>) -> PolarsResult<i64> {
            match time_unit {
                TimeUnit::Nanoseconds => Duration::add_ns,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Milliseconds => Duration::add_ms,
            }
        }
        match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => {
                let adder = adder(time_unit);
                Ok(self
                    .0
                    .try_apply(|t| {
                        roll_forward(
                            t,
                            Some(tz),
                            timestamp_to_datetime,
                            datetime_to_timestamp,
                            adder,
                            &add_one_month,
                            &subtract_one_day,
                        )
                    })?
                    .into_datetime(time_unit, Some(tz.to_string())))
            }
            _ => {
                let adder = adder(time_unit);
                Ok(self
                    .0
                    .try_apply(|t| {
                        roll_forward(
                            t,
                            NO_TIMEZONE,
                            timestamp_to_datetime,
                            datetime_to_timestamp,
                            adder,
                            &add_one_month,
                            &subtract_one_day,
                        )
                    })?
                    .into_datetime(time_unit, None))
            }
        }
    }
}

impl PolarsMonthEnd for DateChunked {
    fn month_end(
        &self,
        _time_unit: Option<TimeUnit>,
        _tz: Option<&impl PolarsTimeZone>,
    ) -> PolarsResult<Self> {
        let add_one_month = Duration::parse("1mo");
        let subtract_one_day = Duration::parse("-1d");
        let no_offset = Duration::parse("0ns");
        let ca = self.truncate(Duration::parse("1mo"), no_offset, NO_TIMEZONE)?;
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        Ok(ca
            .0
            .try_apply(|t| {
                let t = (Duration::add_ms(&add_one_month, MSECS_IN_DAY * t as i64, NO_TIMEZONE)?
                    / MSECS_IN_DAY) as i32;
                let t = (Duration::add_ms(&subtract_one_day, MSECS_IN_DAY * t as i64, NO_TIMEZONE)?
                    / MSECS_IN_DAY) as i32;
                Ok(t)
            })?
            .into_date())
    }
}
