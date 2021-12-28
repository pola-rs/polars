use crate::datatypes::Int64Chunked;
use crate::prelude::{DatetimeChunked, TimeUnit};
pub use polars_time::*;

pub fn date_range(
    start: TimeNanoseconds,
    stop: TimeNanoseconds,
    every: Duration,
    closed: ClosedWindow,
    name: &str,
) -> DatetimeChunked {
    Int64Chunked::new_vec(name, date_range_vec(start, stop, every, closed)).into_datetime(TimeUnit::Nanoseconds, None)
}
