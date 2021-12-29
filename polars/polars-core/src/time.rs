use crate::datatypes::Int64Chunked;
use crate::export::chrono::NaiveDateTime;
use crate::prelude::{DatetimeChunked, TimeUnit};
use polars_time::export::chrono::Datelike;
pub use polars_time::*;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

pub fn date_range(
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
    name: &str,
    tu: TimeUnit,
) -> DatetimeChunked {
    Int64Chunked::new_vec(
        name,
        date_range_vec(start, stop, every, closed, tu.to_polars_time()),
    )
    .into_datetime(tu, None)
}
