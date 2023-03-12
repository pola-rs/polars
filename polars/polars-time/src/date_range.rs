#[cfg(feature = "timezones")]
use arrow::temporal_conversions::{
    parse_offset, timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::{DateTime, LocalResult, TimeZone as TimeZoneTrait, Utc};
use chrono::{Datelike, FixedOffset, NaiveDateTime};
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

#[cfg(feature = "timezones")]
fn localize_timestamp<T: TimeZoneTrait>(timestamp: i64, tu: TimeUnit, tz: T) -> PolarsResult<i64> {
    let ndt = match tu {
        TimeUnit::Nanoseconds => timestamp_ns_to_datetime(timestamp),
        TimeUnit::Microseconds => timestamp_us_to_datetime(timestamp),
        TimeUnit::Milliseconds => timestamp_ms_to_datetime(timestamp),
    };
    let dt: PolarsResult<DateTime<Utc>> = match tz.from_local_datetime(&ndt) {
        LocalResult::Single(dt) => Ok(dt.with_timezone(&Utc)),
        LocalResult::Ambiguous(_, _) => {
            polars_bail!(ComputeError: "ambiguous timestamps are not (yet) supported")
        }
        LocalResult::None => {
            polars_bail!(ComputeError: "non-existent timestamps are not (yet) supported")
        }
    };
    match tu {
        TimeUnit::Nanoseconds => Ok(dt?.timestamp_nanos()),
        TimeUnit::Microseconds => Ok(dt?.timestamp_micros()),
        TimeUnit::Milliseconds => Ok(dt?.timestamp_millis()),
    }
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
    let s = if start > stop {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
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
            date_range_vec(start, stop, every, closed, tu, None::<&FixedOffset>)?,
        )
        .into_datetime(tu, None),
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
