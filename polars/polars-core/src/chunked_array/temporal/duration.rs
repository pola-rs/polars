use crate::export::chrono::Duration as ChronoDuration;
use crate::prelude::DataType::Duration;
use crate::prelude::*;
use arrow::temporal_conversions::{MILLISECONDS, MILLISECONDS_IN_DAY, NANOSECONDS};

const NANOSECONDS_IN_MILLISECOND: i64 = 1_000_000;
const SECONDS_IN_HOUR: i64 = 3600;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match self.2.as_ref().unwrap() {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.2 = Some(Duration(tu))
    }

    pub fn new_from_duration(name: &str, v: &[ChronoDuration], tu: TimeUnit) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: &ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Milliseconds => |v: &ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.iter().map(func).collect_trusted::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_duration(tu)
    }

    /// Extract the hours from a `Duration`
    pub fn hours(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / (MILLISECONDS * SECONDS_IN_HOUR),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_HOUR),
        }
    }

    /// Extract the days from a `Duration`
    pub fn days(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS_IN_DAY,
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_DAY),
        }
    }

    /// Extract the milliseconds from a `Duration`
    pub fn milliseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => self.0.clone(),
            TimeUnit::Nanoseconds => &self.0 / 1_000_000,
        }
    }

    /// Extract the nanoseconds from a `Duration`
    pub fn nanoseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 * NANOSECONDS_IN_MILLISECOND,
            TimeUnit::Nanoseconds => self.0.clone(),
        }
    }

    /// Extract the seconds from a `Duration`
    pub fn seconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS,
            TimeUnit::Nanoseconds => &self.0 / NANOSECONDS,
        }
    }
}
