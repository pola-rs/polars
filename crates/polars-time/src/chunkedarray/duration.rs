use arrow::temporal_conversions::{
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
            TimeUnit::Milliseconds => {
                (&self.0).wrapping_trunc_div_scalar(MILLISECONDS * SECONDS_IN_HOUR)
            },
            TimeUnit::Microseconds => {
                (&self.0).wrapping_trunc_div_scalar(MICROSECONDS * SECONDS_IN_HOUR)
            },
            TimeUnit::Nanoseconds => {
                (&self.0).wrapping_trunc_div_scalar(NANOSECONDS * SECONDS_IN_HOUR)
            },
        }
    }

    /// Extract the days from a `Duration`
    fn days(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => (&self.0).wrapping_trunc_div_scalar(MILLISECONDS_IN_DAY),
            TimeUnit::Microseconds => {
                (&self.0).wrapping_trunc_div_scalar(MICROSECONDS * SECONDS_IN_DAY)
            },
            TimeUnit::Nanoseconds => {
                (&self.0).wrapping_trunc_div_scalar(NANOSECONDS * SECONDS_IN_DAY)
            },
        }
    }

    /// Extract the seconds from a `Duration`
    fn minutes(&self) -> Int64Chunked {
        let tu = match self.time_unit() {
            TimeUnit::Milliseconds => MILLISECONDS,
            TimeUnit::Microseconds => MICROSECONDS,
            TimeUnit::Nanoseconds => NANOSECONDS,
        };
        (&self.0).wrapping_trunc_div_scalar(tu * 60)
    }

    /// Extract the seconds from a `Duration`
    fn seconds(&self) -> Int64Chunked {
        let tu = match self.time_unit() {
            TimeUnit::Milliseconds => MILLISECONDS,
            TimeUnit::Microseconds => MICROSECONDS,
            TimeUnit::Nanoseconds => NANOSECONDS,
        };
        (&self.0).wrapping_trunc_div_scalar(tu)
    }

    /// Extract the milliseconds from a `Duration`
    fn milliseconds(&self) -> Int64Chunked {
        let t = match self.time_unit() {
            TimeUnit::Milliseconds => return self.0.clone(),
            TimeUnit::Microseconds => 1000,
            TimeUnit::Nanoseconds => NANOSECONDS_IN_MILLISECOND,
        };
        (&self.0).wrapping_trunc_div_scalar(t)
    }

    /// Extract the microseconds from a `Duration`
    fn microseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 * 1000,
            TimeUnit::Microseconds => self.0.clone(),
            TimeUnit::Nanoseconds => (&self.0).wrapping_trunc_div_scalar(1000),
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
