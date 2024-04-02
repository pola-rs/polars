use arrow::legacy::time_zone::Tz;
use chrono::NaiveDateTime;
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{
    timestamp_ms_to_datetime, MILLISECONDS, SECONDS_IN_DAY,
};

use crate::period_start::{
    get_timestamp_closures, roll_month_backward, roll_quarter_backward, roll_year_backward,
};
use crate::windows::duration::Duration;

// roll forward to the last day of the month
fn roll_month_forward(
    t: i64,
    time_zone: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
) -> PolarsResult<i64> {
    let t = roll_month_backward(t, time_zone, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = offset_fn(&Duration::parse("1mo"), t, time_zone)?;
    offset_fn(&Duration::parse("-1d"), t, time_zone)
}

// roll forward to the last day of the quarter
fn roll_quarter_forward(
    t: i64,
    time_zone: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
) -> PolarsResult<i64> {
    let t = roll_quarter_backward(t, time_zone, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = offset_fn(&Duration::parse("3mo"), t, time_zone)?;
    offset_fn(&Duration::parse("-1d"), t, time_zone)
}

// roll forward to the last day of the year
fn roll_year_forward(
    t: i64,
    time_zone: Option<&Tz>,
    timestamp_to_datetime: fn(i64) -> NaiveDateTime,
    datetime_to_timestamp: fn(NaiveDateTime) -> i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
) -> PolarsResult<i64> {
    let t = roll_year_backward(t, time_zone, timestamp_to_datetime, datetime_to_timestamp)?;
    let t = offset_fn(&Duration::parse("1y"), t, time_zone)?;
    offset_fn(&Duration::parse("-1d"), t, time_zone)
}

pub trait PolarsDateEnd {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
    fn quarter_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
    fn year_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsDateEnd for DatetimeChunked {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, offset_fn) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply_nonnull_values_generic(|t| {
                roll_month_forward(
                    t,
                    time_zone,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    offset_fn,
                )
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }

    fn quarter_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, offset_fn) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply_nonnull_values_generic(|t| {
                roll_quarter_forward(
                    t,
                    time_zone,
                    timestamp_to_datetime,
                    datetime_to_timestamp,
                    offset_fn,
                )
            })?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }

    fn year_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self> {
        let (timestamp_to_datetime, datetime_to_timestamp, offset_fn) =
            get_timestamp_closures(self.time_unit());
        Ok(self
            .0
            .try_apply_nonnull_values_generic(|t| {
                roll_year_forward(
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

impl PolarsDateEnd for DateChunked {
    fn month_end(&self, _time_zone: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.0.try_apply_nonnull_values_generic(|t| {
            let fwd = roll_month_forward(
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
    fn quarter_end(&self, _time_zone: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.0.try_apply_nonnull_values_generic(|t| {
            let fwd = roll_quarter_forward(
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
    fn year_end(&self, _time_zone: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.0.try_apply_nonnull_values_generic(|t| {
            let fwd = roll_year_forward(
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
