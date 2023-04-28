use chrono::Timelike;
use polars_arrow::export::arrow::temporal_conversions::NANOSECONDS;

use super::*;

const SECONDS_IN_MINUTE: i64 = 60;
const SECONDS_IN_HOUR: i64 = 3_600;

pub(crate) fn time_to_time64ns(time: &NaiveTime) -> i64 {
    (time.hour() as i64 * SECONDS_IN_HOUR
        + time.minute() as i64 * SECONDS_IN_MINUTE
        + time.second() as i64)
        * NANOSECONDS
        + time.nanosecond() as i64
}

pub trait TimeMethods {
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> UInt32Chunked;

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> UInt32Chunked;

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> UInt32Chunked;

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> UInt32Chunked;

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> TimeChunked;
}

impl TimeMethods for TimeChunked {
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_hour)
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_minute)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_second)
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> UInt32Chunked {
        self.apply_kernel_cast::<UInt32Type>(&time_to_nanosecond)
    }

    fn parse_from_str_slice(name: &str, v: &[&str], fmt: &str) -> TimeChunked {
        let mut ca: Int64Chunked = v
            .iter()
            .map(|s| {
                NaiveTime::parse_from_str(s, fmt)
                    .ok()
                    .as_ref()
                    .map(time_to_time64ns)
            })
            .collect_trusted();
        ca.rename(name);
        ca.into()
    }
}
