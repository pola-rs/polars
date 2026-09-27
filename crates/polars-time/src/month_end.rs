use polars_arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::{try_localize_datetime, unlocalize_datetime};
use polars_core::prelude::*;
use polars_core::utils::polars_arrow::temporal_conversions::{MILLISECONDS, SECONDS_IN_DAY};
use polars_defs::time::duration::Duration;

use crate::month_start::roll_backward;

// roll forward to the last day of the month
fn roll_forward(t: i64, time_zone: Option<&Tz>, tu: TimeUnit) -> PolarsResult<i64> {
    // Use Ambiguous::Latest to roll back to the start of the month. It doesn't matter
    // if that timestamp lands on an ambiguous time as we then add 1 month anyway, we
    // could just as well use Ambiguous::Earliest.
    let naive_t = match time_zone {
        #[cfg(feature = "timezones")]
        Some(tz) => tu.datetime_to_timestamp(unlocalize_datetime(tu.timestamp_to_datetime(t), tz)),
        _ => t,
    };
    let naive_month_start_t = roll_backward(naive_t, None, tu)?;
    let naive_result = Duration::parse("-1d").add(
        tu,
        Duration::parse("1mo").add(tu, naive_month_start_t, None)?,
        None,
    )?;
    let result = match time_zone {
        #[cfg(feature = "timezones")]
        Some(tz) => tu.datetime_to_timestamp(
            try_localize_datetime(
                tu.timestamp_to_datetime(naive_result),
                tz,
                Ambiguous::Raise,
                NonExistent::Raise,
            )?
            .expect("we didn't use Ambiguous::Null or NonExistent::Null"),
        ),
        _ => naive_result,
    };
    Ok(result)
}

pub trait PolarsMonthEnd {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl PolarsMonthEnd for DatetimeChunked {
    fn month_end(&self, time_zone: Option<&Tz>) -> PolarsResult<Self> {
        Ok(self
            .phys
            .try_apply_nonnull_values_generic(|t| roll_forward(t, time_zone, self.time_unit()))?
            .into_datetime(self.time_unit(), self.time_zone().clone()))
    }
}

impl PolarsMonthEnd for DateChunked {
    fn month_end(&self, _time_zone: Option<&Tz>) -> PolarsResult<Self> {
        const MSECS_IN_DAY: i64 = MILLISECONDS * SECONDS_IN_DAY;
        let ret = self.phys.try_apply_nonnull_values_generic(|t| {
            let fwd = roll_forward(MSECS_IN_DAY * t as i64, None, TimeUnit::Milliseconds)?;
            PolarsResult::Ok((fwd / MSECS_IN_DAY) as i32)
        })?;
        Ok(ret.into_date())
    }
}
