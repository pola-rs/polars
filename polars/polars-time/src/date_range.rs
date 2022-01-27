use crate::prelude::*;
use chrono::{Datelike, NaiveDateTime};
use polars_core::prelude::*;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

pub fn date_range(
    name: &str,
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
) -> DatetimeChunked {
    Int64Chunked::new_vec(name, date_range_vec(start, stop, every, closed, tu))
        .into_datetime(tu, None)
}
