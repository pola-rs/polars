#[cfg(feature = "timezones")]
use chrono::TimeZone;
use now::DateTimeNow;
use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::*;
use polars_defs::time::duration::Duration;
use polars_defs::time::group_by::{ClosedWindow, StartBy};

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
    tu: TimeUnit,
    period: Duration,
    mut start: i64,
    closed_window: ClosedWindow,
    tz: Option<&Tz>,
) -> PolarsResult<Bounds> {
    every.negative = !every.negative;
    let mut stop = period.add(tu, start, tz)?;

    while Bounds::new(start, stop).is_past(t, closed_window) {
        let mut gap = start - t;
        if matches!(closed_window, ClosedWindow::Right | ClosedWindow::None) {
            gap += 1;
        }
        debug_assert!(gap >= 1);

        // Ceil division
        let stride = (gap + every.nte_duration(tu) - 1) / every.nte_duration(tu);
        debug_assert!(stride >= 1);
        let stride = std::cmp::max(stride, 1);

        start = (every * stride).add(tu, start, tz)?;
        stop = period.add(tu, start, tz)?;
    }
    Ok(Bounds::new_checked(start, stop))
}

/// Represents a window in time
#[derive(Copy, Clone)]
pub struct Window {
    // The ith window start is expressed via this equation:
    //   window_start_i = zero + every * i
    //   window_stop_i = zero + every * i + period
    pub(crate) every: Duration,
    pub(crate) period: Duration,
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

    /// Truncate the given timestamp in `tu` by the window boundary.
    pub fn truncate(&self, tu: TimeUnit, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        self.every.truncate(tu, t, tz)
    }

    /// Round the given timestamp in `tu` by the window boundary.
    pub fn round(&self, tu: TimeUnit, t: i64, tz: Option<&Tz>) -> PolarsResult<i64> {
        let t = t + self.every.duration(tu) / 2;
        self.truncate(tu, t, tz)
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    pub fn get_earliest_bounds(
        &self,
        tu: TimeUnit,
        t: i64,
        closed_window: ClosedWindow,
        tz: Option<&Tz>,
    ) -> PolarsResult<Bounds> {
        let start = self.truncate(tu, t, tz)?;
        let start = self.offset.add(tu, start, tz)?;
        ensure_t_in_or_in_front_of_window(self.every, t, tu, self.period, start, closed_window, tz)
    }

    pub(crate) fn estimate_overlapping_bounds(&self, tu: TimeUnit, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration(tu)
            + self.period.duration(tu) / self.every.duration(tu)) as usize
    }

    pub fn get_overlapping_bounds_iter<'a>(
        &'a self,
        boundary: Bounds,
        closed_window: ClosedWindow,
        tu: TimeUnit,
        tz: Option<&'a Tz>,
        start_by: StartBy,
    ) -> PolarsResult<BoundsIter<'a>> {
        BoundsIter::new(*self, closed_window, boundary, tu, tz, start_by)
    }

    /// The start of the first window for data whose first value is `t0`, as placed by
    /// `start_by`, `every` and `offset`.
    pub fn first_window_start(
        &self,
        t0: i64,
        closed_window: ClosedWindow,
        tu: TimeUnit,
        tz: Option<&Tz>,
        start_by: StartBy,
    ) -> PolarsResult<i64> {
        match start_by {
            StartBy::DataPoint => Ok(t0),
            StartBy::WindowBound => Ok(self.get_earliest_bounds(tu, t0, closed_window, tz)?.start),
            _ => {
                // Find the beginning of the week in the time zone, then place the window
                // start on the requested weekday plus `offset`.
                let dt = tu.timestamp_to_datetime(t0);
                let (week_start, tz) = match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => (
                        tz.from_utc_datetime(&dt).beginning_of_week().naive_utc(),
                        Some(tz),
                    ),
                    _ => (dt.and_utc().beginning_of_week().naive_utc(), None),
                };
                let start = tu.datetime_to_timestamp(week_start);
                let start = Duration::parse(&format!("{}d", start_by.weekday().unwrap()))
                    .add(tu, start, tz)?;
                let start = self.offset.add(tu, start, tz)?;
                // Make sure the first datapoint has a chance to be included.
                let bounds = ensure_t_in_or_in_front_of_window(
                    self.every,
                    t0,
                    tu,
                    self.period,
                    start,
                    closed_window,
                    tz,
                )?;
                Ok(bounds.start)
            },
        }
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
        let start = window.first_window_start(boundary.start, closed_window, tu, tz, start_by)?;
        let stop = window.period.add(tu, start, tz)?;
        Ok(Self {
            window,
            boundary,
            bi: Bounds::new(start, stop),
            tu,
            tz,
        })
    }
}

impl Iterator for BoundsIter<'_> {
    type Item = Bounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bi.start < self.boundary.stop {
            let out = self.bi;
            // TODO: find some way to propagate error instead of unwrapping?
            // Issue is that `next` needs to return `Option`.
            self.bi.start = self
                .window
                .every
                .add(self.tu, self.bi.start, self.tz)
                .unwrap();
            self.bi.stop = self
                .window
                .period
                .add(self.tu, self.bi.start, self.tz)
                .unwrap();
            Some(out)
        } else {
            None
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let n: i64 = n.try_into().unwrap();
        if self.bi.start < self.boundary.stop {
            self.bi.start = (self.window.every * n)
                .add(self.tu, self.bi.start, self.tz)
                .unwrap();
            self.bi.stop = self
                .window
                .period
                .add(self.tu, self.bi.start, self.tz)
                .unwrap();
            self.next()
        } else {
            None
        }
    }
}

impl<'a> BoundsIter<'a> {
    /// Number of iterations to advance, such that the bounds are on target; or, in
    /// the case of non-constant duration, close to target.
    /// Follows the `nth()` convention on Iterator indexing, i.e., a return value of 0
    /// implies advancing 1 iteration.
    pub fn get_stride(&self, target: i64) -> usize {
        let mut stride = 0;
        if self.bi.start < self.boundary.stop && target > self.bi.start {
            let gap = target - self.bi.start;
            let every = self.window.every.nte_duration(self.tu);
            let period = self.window.period.nte_duration(self.tu);
            if gap > every + period {
                stride = ((gap - period) as usize) / (every as usize);
            }
        }
        stride
    }
}
