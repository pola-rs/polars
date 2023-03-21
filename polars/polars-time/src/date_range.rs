#[cfg(feature = "timezones")]
use arrow::temporal_conversions::parse_offset;
use chrono::{Datelike, NaiveDateTime};
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;
#[cfg(feature = "timezones")]
use crate::utils::localize_timestamp;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

#[cfg(feature = "private")]
#[doc(hidden)]
pub fn date_range_impl(
    name: &str,
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    _tz: Option<&TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    if start > stop {
        polars_bail!(ComputeError: "'start' cannot be greater than 'stop'")
    }
    if every.negative {
        polars_bail!(ComputeError: "'interval' cannot be negative")
    }
    let mut out = match _tz {
        #[cfg(feature = "timezones")]
        Some(tz) => match tz.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let start = localize_timestamp(start, tu, tz);
                let stop = localize_timestamp(stop, tu, tz);
                Int64Chunked::new_vec(
                    name,
                    date_range_vec(start?, stop?, every, closed, tu, Some(&tz))?,
                )
                .into_datetime(tu, _tz.cloned())
            }
            Err(_) => match parse_offset(tz) {
                Ok(tz) => {
                    let start = localize_timestamp(start, tu, tz);
                    let stop = localize_timestamp(stop, tu, tz);
                    Int64Chunked::new_vec(
                        name,
                        date_range_vec(start?, stop?, every, closed, tu, Some(&tz))?,
                    )
                    .into_datetime(tu, _tz.cloned())
                }
                _ => polars_bail!(ComputeError: "unable to parse time zone: {}", tz),
            },
        },
        _ => Int64Chunked::new_vec(
            name,
            date_range_vec(start, stop, every, closed, tu, NO_TIMEZONE)?,
        )
        .into_datetime(tu, None),
    };

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`DatetimeChunked`] from a given `start` and `stop` date and a given `every` interval.
pub fn date_range(
    name: &str,
    start: NaiveDateTime,
    stop: NaiveDateTime,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let (start, stop) = match tu {
        TimeUnit::Nanoseconds => (start.timestamp_nanos(), stop.timestamp_nanos()),
        TimeUnit::Microseconds => (
            start.timestamp() + start.timestamp_subsec_micros() as i64,
            stop.timestamp() + stop.timestamp_subsec_millis() as i64,
        ),
        TimeUnit::Milliseconds => (start.timestamp_millis(), stop.timestamp_millis()),
    };
    date_range_impl(name, start, stop, every, closed, tu, tz.as_ref())
}
