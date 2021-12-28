use crate::datatypes::Int64Chunked;
use crate::prelude::{DatetimeChunked, TimeUnit};
pub use polars_time::*;
use polars_time::export::chrono::Datelike;
use crate::export::chrono::NaiveDateTime;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

pub fn date_range(
    start: TimeNanoseconds,
    stop: TimeNanoseconds,
    every: Duration,
    closed: ClosedWindow,
    name: &str,
) -> DatetimeChunked {
    Int64Chunked::new_vec(name, date_range_vec(start, stop, every, closed)).into_datetime(TimeUnit::Nanoseconds, None)
}
