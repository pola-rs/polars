use polars_arrow::export::arrow::temporal_conversions::{
    MICROSECONDS, MILLISECONDS, MILLISECONDS_IN_DAY, NANOSECONDS, SECONDS_IN_DAY,
};

use super::*;

const NANOSECONDS_IN_MILLISECOND: i64 = 1_000_000;
const SECONDS_IN_HOUR: i64 = 3600;

pub trait DurationMethods {
    /// Extract the hours from a `Duration`
    fn hours(&self) -> Int64Chunked;

    /// Extract the days from a `Duration`
    fn days(&self) -> Int64Chunked;

    /// Extract the minutes from a `Duration`
    fn minutes(&self) -> Int64Chunked;

    /// Extract the seconds from a `Duration`
    fn seconds(&self) -> Int64Chunked;

    /// Extract the milliseconds from a `Duration`
    fn milliseconds(&self) -> Int64Chunked;

    /// Extract the microseconds from a `Duration`
    fn microseconds(&self) -> Int64Chunked;

    /// Extract the nanoseconds from a `Duration`
    fn nanoseconds(&self) -> Int64Chunked;
}

impl DurationMethods for DurationChunked {
    /// Extract the hours from a `Duration`
    fn hours(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / (MILLISECONDS * SECONDS_IN_HOUR),
            TimeUnit::Microseconds => &self.0 / (MICROSECONDS * SECONDS_IN_HOUR),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_HOUR),
        }
    }

    /// Extract the days from a `Duration`
    fn days(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS_IN_DAY,
            TimeUnit::Microseconds => &self.0 / (MICROSECONDS * SECONDS_IN_DAY),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_DAY),
        }
    }

    /// Extract the seconds from a `Duration`
    fn minutes(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / (MILLISECONDS * 60),
            TimeUnit::Microseconds => &self.0 / (MICROSECONDS * 60),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * 60),
        }
    }

    /// Extract the seconds from a `Duration`
    fn seconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS,
            TimeUnit::Microseconds => &self.0 / MICROSECONDS,
            TimeUnit::Nanoseconds => &self.0 / NANOSECONDS,
        }
    }

    /// Extract the milliseconds from a `Duration`
    fn milliseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => self.0.clone(),
            TimeUnit::Microseconds => self.0.clone() / 1000,
            TimeUnit::Nanoseconds => &self.0 / NANOSECONDS_IN_MILLISECOND,
        }
    }

    /// Extract the microseconds from a `Duration`
    fn microseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 * 1000,
            TimeUnit::Microseconds => self.0.clone(),
            TimeUnit::Nanoseconds => &self.0 / 1000,
        }
    }

    /// Extract the nanoseconds from a `Duration`
    fn nanoseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 * NANOSECONDS_IN_MILLISECOND,
            TimeUnit::Microseconds => &self.0 * 1000,
            TimeUnit::Nanoseconds => self.0.clone(),
        }
    }
}
