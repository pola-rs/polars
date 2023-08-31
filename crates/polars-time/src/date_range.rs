use chrono::{Datelike, NaiveDateTime, NaiveTime};
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;
#[cfg(feature = "timezones")]
use crate::utils::localize_timestamp;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

#[doc(hidden)]
pub fn date_range_impl(
    name: &str,
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    _tz: Option<&TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let mut out = match _tz {
        #[cfg(feature = "timezones")]
        Some(tz) => match tz.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let start = localize_timestamp(start, tu, tz);
                let end = localize_timestamp(end, tu, tz);
                Int64Chunked::new_vec(
                    name,
                    temporal_range_vec(start?, end?, interval, closed, tu, Some(&tz))?,
                )
                .into_datetime(tu, _tz.cloned())
            },
            Err(_) => polars_bail!(ComputeError: "unable to parse time zone: '{}'", tz),
        },
        _ => Int64Chunked::new_vec(
            name,
            temporal_range_vec(start, end, interval, closed, tu, None)?,
        )
        .into_datetime(tu, None),
    };

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`DatetimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn date_range(
    name: &str,
    start: NaiveDateTime,
    end: NaiveDateTime,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let (start, end) = match tu {
        TimeUnit::Nanoseconds => (start.timestamp_nanos(), end.timestamp_nanos()),
        TimeUnit::Microseconds => (start.timestamp_micros(), end.timestamp_micros()),
        TimeUnit::Milliseconds => (start.timestamp_millis(), end.timestamp_millis()),
    };
    date_range_impl(name, start, end, interval, closed, tu, tz.as_ref())
}

#[doc(hidden)]
pub fn time_range_impl(
    name: &str,
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    let mut out = Int64Chunked::new_vec(
        name,
        temporal_range_vec(start, end, interval, closed, TimeUnit::Nanoseconds, None)?,
    )
    .into_time();

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`TimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn time_range(
    name: &str,
    start: NaiveTime,
    end: NaiveTime,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    let start = time_to_time64ns(&start);
    let end = time_to_time64ns(&end);
    time_range_impl(name, start, end, interval, closed)
}
