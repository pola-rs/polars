use arrow::legacy::time_zone::Tz;
use arrow::temporal_conversions::*;
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
use now::DateTimeNow;
use polars_core::prelude::*;

use crate::prelude::*;

/// Ensure that earliest datapoint (`t`) is in, or in front of, first window.
///
/// For example, if we have:
///
/// - first datapoint is `2020-01-01 01:00`
/// - `every` is `'1d'`
/// - `period` is `'2d'`
/// - `offset` is `'6h'`
///
/// then truncating the earliest datapoint by `every` and adding `offset` results
/// in the window `[2020-01-01 06:00, 2020-01-03 06:00)`. To give the earliest datapoint
/// a chance of being included, we then shift the window back by `every` to
/// `[2019-12-31 06:00, 2020-01-02 06:00)`.
pub(crate) fn ensure_t_in_or_in_front_of_window(
    mut every: Duration,
    t: i64,
    offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
    period: Duration,
    mut start: i64,
    closed_window: ClosedWindow,
    tz: Option<&Tz>,
) -> PolarsResult<Bounds> {
    every.negative = !every.negative;
    let mut stop = offset_fn(&period, start, tz)?;
    while Bounds::new(start, stop).is_past(t, closed_window) {
        start = offset_fn(&every, start, tz)?;
        stop = offset_fn(&period, start, tz)?;
    }
    Ok(Bounds::new_checked(start, stop))
}

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
        debug_assert!(!every.negative);
        Self {
            every,
            period,
            offset,
        }
    }

    /// Truncate the given ns timestamp by the window boundary.
    pub fn truncate_ns(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_ns(t, tz)
    }

    /// Truncate the given us timestamp by the window boundary.
    pub fn truncate_us(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_us(t, tz)
    }

    /// Truncate the given ms timestamp by the window boundary.
    pub fn truncate_ms(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate_ms(t, tz)
    }

    /// Round the given ns timestamp by the window boundary.
    pub fn round_ns(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns() / 2_i64;
        self.truncate_ns(t, tz)
    }

    /// Round the given us timestamp by the window boundary.
    pub fn round_us(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Microsecond) as i64);
        self.truncate_us(t, tz)
    }

    /// Round the given ms timestamp by the window boundary.
    pub fn round_ms(&self, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration_ns()
            / (2 * timeunit_scale(ArrowTimeUnit::Nanosecond, ArrowTimeUnit::Millisecond) as i64);
        self.truncate_ms(t, tz)
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    pub fn get_earliest_bounds_ns(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_ns(t, tz)?;
        let start = self.offset.add_ns(start, tz)?;
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            Duration::add_ns,
            self.period,
            start,
            closed_window,
            tz,
        )
    }

    pub fn get_earliest_bounds_us(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_us(t, tz)?;
        let start = self.offset.add_us(start, tz)?;
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            Duration::add_us,
            self.period,
            start,
            closed_window,
            tz,
        )
    }

    pub fn get_earliest_bounds_ms(
        &self,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate_ms(t, tz)?;
        let start = self.offset.add_ms(start, tz)?;
        ensure_t_in_or_in_front_of_window(
            self.every,
            t,
            Duration::add_ms,
            self.period,
            start,
            closed_window,
            tz,
        )
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

    pub fn get_overlapping_bounds_iter<'a>(
        &'a self,
        boundary: Bounds,
        closed_window: ClosedWindow,
        tu: TimeUnit,
        tz: Option<&'a Tz>,
        start_by: StartBy,
    ) -> PolarsResult<BoundsIter> {
        BoundsIter::new(*self, closed_window, boundary, tu, tz, start_by)
    }
}

pub struct BoundsIter<'a> {
    window: Window,
    // wrapping boundary
    boundary: Bounds,
    // boundary per window iterator
    bi: Bounds,
    tu: TimeUnit,
    tz: Option<&'a Tz>,
}
impl<'a> BoundsIter<'a> {
    fn new(
        window: Window,
        closed_window: ClosedWindow,
        boundary: Bounds,
        tu: TimeUnit,
        tz: Option<&'a Tz>,
        start_by: StartBy,
    ) -> PolarsResult<Self> {
        let bi = match start_by {
            StartBy::DataPoint => {
                let mut boundary = boundary;
                let offset_fn = match tu {
                    TimeUnit::Nanoseconds => Duration::add_ns,
                    TimeUnit::Microseconds => Duration::add_us,
                    TimeUnit::Milliseconds => Duration::add_ms,
                };
                boundary.stop = offset_fn(&window.period, boundary.start, tz)?;
                boundary
            },
            StartBy::WindowBound => match tu {
                TimeUnit::Nanoseconds => {
                    window.get_earliest_bounds_ns(boundary.start, closed_window, tz)?
                },
                TimeUnit::Microseconds => {
                    window.get_earliest_bounds_us(boundary.start, closed_window, tz)?
                },
                TimeUnit::Milliseconds => {
                    window.get_earliest_bounds_ms(boundary.start, closed_window, tz)?
                },
            },
            _ => {
                {
                    #[allow(clippy::type_complexity)]
                    let (from, to, offset_fn): (
                        fn(i64) -> NaiveDateTime,
                        fn(NaiveDateTime) -> i64,
                        fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>,
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
                    let dt = from(boundary.start);
                    match tz {
                        #[cfg(feature = "timezones")]
                        Some(tz) => {
                            let dt = tz.from_utc_datetime(&dt);
                            let dt = dt.beginning_of_week();
                            let dt = dt.naive_utc();
                            let start = to(dt);
                            // adjust start of the week based on given day of the week
                            let start = offset_fn(
                                &Duration::parse(&format!("{}d", start_by.weekday().unwrap())),
                                start,
                                Some(tz),
                            )?;
                            // apply the 'offset'
                            let start = offset_fn(&window.offset, start, Some(tz))?;
                            // make sure the first datapoint has a chance to be included
                            // and compute the end of the window defined by the 'period'
                            ensure_t_in_or_in_front_of_window(
                                window.every,
                                boundary.start,
                                offset_fn,
                                window.period,
                                start,
                                closed_window,
                                Some(tz),
                            )?
                        },
                        _ => {
                            let tz = chrono::Utc;
                            let dt = dt.and_local_timezone(tz).unwrap();
                            let dt = dt.beginning_of_week();
                            let dt = dt.naive_utc();
                            let start = to(dt);
                            // adjust start of the week based on given day of the week
                            let start = offset_fn(
                                &Duration::parse(&format!("{}d", start_by.weekday().unwrap())),
                                start,
                                None,
                            )
                            .unwrap();
                            // apply the 'offset'
                            let start = offset_fn(&window.offset, start, None).unwrap();
                            // make sure the first datapoint has a chance to be included
                            // and compute the end of the window defined by the 'period'
                            ensure_t_in_or_in_front_of_window(
                                window.every,
                                boundary.start,
                                offset_fn,
                                window.period,
                                start,
                                closed_window,
                                None,
                            )?
                        },
                    }
                }
            },
        };
        Ok(Self {
            window,
            boundary,
            bi,
            tu,
            tz,
        })
    }
}

impl<'a> Iterator for BoundsIter<'a> {
    type Item = Bounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bi.start < self.boundary.stop {
            let out = self.bi;
            match self.tu {
                // TODO: find some way to propagate error instead of unwrapping?
                // Issue is that `next` needs to return `Option`.
                TimeUnit::Nanoseconds => {
                    self.bi.start = self.window.every.add_ns(self.bi.start, self.tz).unwrap();
                    self.bi.stop = self.window.every.add_ns(self.bi.stop, self.tz).unwrap();
                },
                TimeUnit::Microseconds => {
                    self.bi.start = self.window.every.add_us(self.bi.start, self.tz).unwrap();
                    self.bi.stop = self.window.every.add_us(self.bi.stop, self.tz).unwrap();
                },
                TimeUnit::Milliseconds => {
                    self.bi.start = self.window.every.add_ms(self.bi.start, self.tz).unwrap();
                    self.bi.stop = self.window.every.add_ms(self.bi.stop, self.tz).unwrap();
                },
            }
            Some(out)
        } else {
            None
        }
    }
}
