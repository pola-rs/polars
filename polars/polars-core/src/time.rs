use crate::datatypes::Int64Chunked;
use crate::prelude::DatetimeChunked;
pub use polars_time::*;

pub fn date_range(
    start: TimeNanoseconds,
    stop: TimeNanoseconds,
    every: Duration,
    closed: ClosedWindow,
    name: &str,
) -> DatetimeChunked {
    Int64Chunked::new_vec(name, date_range_vec(start, stop, every, closed)).into_date()
}
