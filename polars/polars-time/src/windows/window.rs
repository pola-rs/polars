#[cfg(feature = "timezones")]
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use now::DateTimeNow;
use polars_arrow::export::arrow::temporal_conversions::*;
use polars_core::prelude::*;
use polars_core::utils::arrow::temporal_conversions::{timeunit_scale, SECONDS_IN_DAY};

use crate::prelude::*;

/// Represents a window in time
#[derive(Copy, Clone)]
pub struct Window {
    // The ith window start is expressed via this equation:
    //   window_start_i = zero + every * i
    //   window_stop_i = zero + every * i + period
    every: Duration,
    period: Duration,
    pub offset: Duration,
}

impl Window {
    pub fn new(every: Duration, period: Duration, offset: Duration) -> Self {
        Self {
            every,
            period,
            offset,
        }
    }

    /// Truncate the given ns timestamp by the window boundary.
    pub fn truncate_ns(&self, t: i64) -> i64 {
        let t = self.every.truncate_ns(t);
        self.offset.add_ns(t)
    }

    pub fn truncate_no_offset_ns(&self, t: i64) -> i64 {
        self.every.truncate_ns(t)
    }

    /// Truncate the given ns timestamp by the window boundary.
    pub fn truncate_us(&self, t: i64) -> i64 {
        let t = self.every.truncate_us(t);
        self.offset.add_us(t)
    }

    pub fn truncate_no_offset_us(&self, t: i64) -> i64 {
        self.every.truncate_us(t)
    }

    pub fn truncate_ms(&self, t: i64) -> i64 {
        let t = self.every.truncate_ms(t);
        self.offset.add_ms(t)
    }

    #[inline]
    pub fn truncate_no_offset_ms(&self, t: i64) -> i64 {
        self.every.truncate_ms(t)
    }

    /// Round the given ns timestamp by the window boundary.
    pub fn round_ns(&self, t: i64) -> i64 {
        let t = t + self.every.duration_ns() / 2_i64;
        self.truncate_ns(t)
    }

    /// Round the given us timestamp by the window boundary.
    pub fn round_us(&self, t: i64) -> i64 {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Microsecond) as i64);
        self.truncate_us(t)
    }

    /// Round the given ms timestamp by the window boundary.
    pub fn round_ms(&self, t: i64) -> i64 {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Millisecond) as i64);
        self.truncate_ms(t)
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    ///
    /// For `every` larger than `1day` we just take the given timestamp `t` as start as truncation
    /// does not seems intuitive.
    /// Below 1 day, it make sense to truncate to:
    /// - days
    /// - hours
    /// - 15 minutes
    /// - etc.
    ///
    /// But for 2w3d, it does not make sense to start it on a different lower bound, so we start at `t`
    pub fn get_earliest_bounds_ns(&self, t: i64) -> Bounds {
        let start = if !self.every.months_only()
            && self.every.duration_ns() > NANOSECONDS * SECONDS_IN_DAY
        {
            self.offset.add_ns(t)
        } else {
            // offset is translated in the truncate
            self.truncate_ns(t)
        };

        let stop = self.period.add_ns(start);

        Bounds::new_checked(start, stop)
    }

    pub fn get_earliest_bounds_us(&self, t: i64) -> Bounds {
        let start = if !self.every.months_only()
            && self.every.duration_us() > MICROSECONDS * SECONDS_IN_DAY
        {
            self.offset.add_us(t)
        } else {
            self.truncate_us(t)
        };
        let stop = self.period.add_us(start);

        Bounds::new_checked(start, stop)
    }

    pub fn get_earliest_bounds_ms(&self, t: i64) -> Bounds {
        let start = if !self.every.months_only()
            && self.every.duration_ms() > MILLISECONDS * SECONDS_IN_DAY
        {
            self.offset.add_ms(t)
        } else {
            self.truncate_ms(t)
        };

        let stop = self.period.add_ms(start);

        Bounds::new_checked(start, stop)
    }

    pub(crate) fn estimate_overlapping_bounds_ns(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ns()
            + self.period.duration_ns() / self.every.duration_ns()) as usize
    }

    pub(crate) fn estimate_overlapping_bounds_us(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_us()
            + self.period.duration_us() / self.every.duration_us()) as usize
    }

    pub(crate) fn estimate_overlapping_bounds_ms(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ms()
            + self.period.duration_ms() / self.every.duration_ms()) as usize
    }

    pub fn get_overlapping_bounds_iter(
        &self,
        boundary: Bounds,
        tu: TimeUnit,
        start_by: StartBy,
    ) -> BoundsIter {
        BoundsIter::new(*self, boundary, tu, start_by)
    }
}

pub struct BoundsIter {
    window: Window,
    // wrapping boundary
    boundary: Bounds,
    // boundary per window iterator
    bi: Bounds,
    tu: TimeUnit,
}
impl BoundsIter {
    fn new(window: Window, boundary: Bounds, tu: TimeUnit, start_by: StartBy) -> Self {
        let bi = match start_by {
            StartBy::DataPoint => {
                let mut boundary = boundary;
                let offset_fn = match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                };
                boundary.stop = offset_fn(&window.period, boundary.start);
                boundary
            }
            StartBy::WindowBound => match tu {
                TimeUnit::Nanoseconds => window.get_earliest_bounds_ns(boundary.start),
                TimeUnit::Microseconds => window.get_earliest_bounds_us(boundary.start),
                TimeUnit::Milliseconds => window.get_earliest_bounds_ms(boundary.start),
            },
            StartBy::Monday => {
                #[cfg(feature = "timezones")]
                {
                    #[allow(clippy::type_complexity)]
                    let (from, to, offset): (
                        fn(i64) -> NaiveDateTime,
                        fn(NaiveDateTime) -> i64,
                        fn(&Duration, i64) -> i64,
                    ) = match tu {
                        TimeUnit::Nanoseconds => (
                            timestamp_ns_to_datetime,
                            datetime_to_timestamp_ns,
                            Duration::add_ns,
                        ),
                        TimeUnit::Microseconds => (
                            timestamp_us_to_datetime,
                            datetime_to_timestamp_us,
                            Duration::add_us,
                        ),
                        TimeUnit::Milliseconds => (
                            timestamp_ms_to_datetime,
                            datetime_to_timestamp_ms,
                            Duration::add_ms,
                        ),
                    };
                    // find beginning of the week.
                    let mut boundary = boundary;
                    let dt = from(boundary.start);
                    let tz = chrono_tz::UTC;
                    let dt = dt.and_local_timezone(tz).unwrap();
                    let dt = dt.beginning_of_week();
                    let dt = dt.naive_utc();
                    let start = to(dt);
                    // apply the 'offset'
                    let start = offset(&window.offset, start);
                    // and compute the end of the window defined by the 'period'
                    let stop = offset(&window.period, start);
                    boundary.start = start;
                    boundary.stop = stop;
                    boundary
                }
                #[cfg(not(feature = "timezones"))]
                {
                    panic!("activate 'timezones' feature")
                }
            }
        };
        Self {
            window,
            boundary,
            bi,
            tu,
        }
    }
}

impl Iterator for BoundsIter {
    type Item = Bounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bi.start < self.boundary.stop {
            let out = self.bi;
            match self.tu {
                TimeUnit::Nanoseconds => {
                    self.bi.start = self.window.every.add_ns(self.bi.start);
                    self.bi.stop = self.window.every.add_ns(self.bi.stop);
                }
                TimeUnit::Microseconds => {
                    self.bi.start = self.window.every.add_us(self.bi.start);
                    self.bi.stop = self.window.every.add_us(self.bi.stop);
                }
                TimeUnit::Milliseconds => {
                    self.bi.start = self.window.every.add_ms(self.bi.start);
                    self.bi.stop = self.window.every.add_ms(self.bi.stop);
                }
            }
            Some(out)
        } else {
            None
        }
    }
}
