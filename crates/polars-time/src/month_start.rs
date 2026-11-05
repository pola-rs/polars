use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars_arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::{try_localize_datetime, unlocalize_datetime};
use polars_core::prelude::*;
use polars_core::utils::polars_arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};

// roll backward to the first day of the month
pub(crate) fn roll_backward(t: i64, tz: Option<&Tz>, tu: TimeUnit) -> PolarsResult<i64> {
    let ts = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => unlocalize_datetime(tu.timestamp_to_datetime(t), tz),
        _ => tu.timestamp_to_datetime(t),
    };
    let date = NaiveDate::from_ymd_opt(ts.year(), ts.month(), 1).ok_or_else(|| {
        polars_err!(
            ComputeError: "Could not construct date {}-{}-1", ts.year(), ts.month()
        )
    })?;
    let time = NaiveTime::from_hms_nano_opt(
        ts.hour(),
        ts.minute(),
        ts.second(),
        ts.and_utc().timestamp_subsec_nanos(),
    )
    .ok_or_else(|| {
        polars_err!(
            ComputeError:
                "Could not construct time {}:{}:{}.{}",
                ts.hour(),
                ts.minute(),
                ts.second(),
                ts.and_utc().timestamp_subsec_nanos()
        )
    })?;
    let ndt = NaiveDateTime::new(date, time);
    let t = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => tu.datetime_to_timestamp(
            try_localize_datetime(ndt, tz, Ambiguous::Raise, NonExistent::Raise)?
                .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
        ),
        _ => tu.datetime_to_timestamp(ndt),
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
        Ok(self
            .phys
            .try_apply_nonnull_values_generic(|t| roll_backward(t, tz, self.time_unit()))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsMonthStart for DateChunked {
    fn month_start(&self, _tz: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.phys.try_apply_nonnull_values_generic(|t| {
            let bwd = roll_backward(MSECS_IN_DAY * t as i64, None, TimeUnit::Milliseconds)?;
            PolarsResult::Ok((bwd / MSECS_IN_DAY) as i32)
        })?;
        Ok(ret.into_date())
    }
}
