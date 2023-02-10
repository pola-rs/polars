use chrono::{Datelike, NaiveDateTime};
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;

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
    let mut out = Int64Chunked::new_vec(name, date_range_vec(start, stop, every, closed, tu))
        .into_datetime(tu, None);

    #[cfg(feature = "timezones")]
    if let Some(tz) = _tz {
        out = out.replace_time_zone(Some(tz))?
    }
    let s = if start > stop {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    out.set_sorted_flag(s);
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
