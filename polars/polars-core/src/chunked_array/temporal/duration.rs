use crate::export::chrono::Duration as ChronoDuration;
use crate::prelude::DataType::Duration;
use crate::prelude::*;
use arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, MILLISECONDS_IN_DAY, NANOSECONDS};

const NANOSECONDS_IN_MILLISECOND: i64 = 1_000_000;
const SECONDS_IN_HOUR: i64 = 3600;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match self.2.as_ref().unwrap() {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    #[must_use]
    pub fn cast_time_unit(&self, tu: TimeUnit) -> Self {
        let current_unit = self.time_unit();
        let mut out = self.clone();
        out.set_time_unit(tu);

        use TimeUnit::*;
        match (current_unit, tu) {
            (Nanoseconds, Microseconds) => {
                let ca = &self.0 / 1_000;
                out.0 = ca;
                out
            }
            (Nanoseconds, Milliseconds) => {
                let ca = &self.0 / 1_000_000;
                out.0 = ca;
                out
            }
            (Microseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            }
            (Microseconds, Milliseconds) => {
                let ca = &self.0 / 1_000;
                out.0 = ca;
                out
            }
            (Milliseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000_000;
                out.0 = ca;
                out
            }
            (Milliseconds, Microseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            }
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => out,
        }
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.2 = Some(Duration(tu))
    }

    /// Construct a new [`DurationChunked`] from an iterator over [`ChronoDuration`].
    pub fn from_duration<I: IntoIterator<Item = ChronoDuration>>(
        name: &str,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(func).collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_duration(tu)
    }

    /// Construct a new [`DurationChunked`] from an iterator over optional [`ChronoDuration`].
    pub fn from_duration_options<I: IntoIterator<Item = Option<ChronoDuration>>>(
        name: &str,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(|opt| opt.map(func));
        Int64Chunked::from_iter_options(name, vals).into_duration(tu)
    }

    /// Extract the hours from a `Duration`
    pub fn hours(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / (MILLISECONDS * SECONDS_IN_HOUR),
            TimeUnit::Microseconds => &self.0 / (MICROSECONDS * SECONDS_IN_HOUR),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_HOUR),
        }
    }

    /// Extract the days from a `Duration`
    pub fn days(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS_IN_DAY,
            TimeUnit::Microseconds => &self.0 / (MICROSECONDS * SECONDS_IN_DAY),
            TimeUnit::Nanoseconds => &self.0 / (NANOSECONDS * SECONDS_IN_DAY),
        }
    }

    /// Extract the milliseconds from a `Duration`
    pub fn milliseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => self.0.clone(),
            TimeUnit::Microseconds => self.0.clone() / 1000,
            TimeUnit::Nanoseconds => &self.0 / 1_000_000,
        }
    }

    /// Extract the nanoseconds from a `Duration`
    pub fn nanoseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 * NANOSECONDS_IN_MILLISECOND,
            TimeUnit::Microseconds => &self.0 * 1000,
            TimeUnit::Nanoseconds => self.0.clone(),
        }
    }

    /// Extract the seconds from a `Duration`
    pub fn seconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.0 / MILLISECONDS,
            TimeUnit::Microseconds => &self.0 / MICROSECONDS,
            TimeUnit::Nanoseconds => &self.0 / NANOSECONDS,
        }
    }
}
