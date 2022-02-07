use crate::prelude::*;
use polars_core::prelude::*;

/// Represents a window in time
#[derive(Copy, Clone)]
pub struct Window {
    // The ith window start is expressed via this equation:
    //   window_start_i = zero + every * i
    //   window_stop_i = zero + every * i + period
    every: Duration,
    period: Duration,
    offset: Duration,
}

impl Window {
    pub fn new(every: Duration, period: Duration, offset: Duration) -> Self {
        Self {
            every,
            period,
            offset,
        }
    }

    #[inline]
    pub fn truncate_ns(&self, t: i64) -> i64 {
        let t = self.every.truncate_ns(t);
        self.offset.add_ns(t)
    }

    #[inline]
    pub fn truncate_no_offset_ns(&self, t: i64) -> i64 {
        self.every.truncate_ns(t)
    }

    #[inline]
    pub fn truncate_ms(&self, t: i64) -> i64 {
        let t = self.every.truncate_ms(t);
        self.offset.add_ms(t)
    }

    #[inline]
    pub fn truncate_no_offset_ms(&self, t: i64) -> i64 {
        self.every.truncate_ms(t)
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    pub fn get_earliest_bounds_ns(&self, t: i64) -> Bounds {
        // original code translates offset here
        // we don't. Seems unintuitive to me.
        let start = self.truncate_ns(t);
        let stop = self.period.add_ns(start);

        Bounds::new_checked(start, stop)
    }

    pub fn get_earliest_bounds_ms(&self, t: i64) -> Bounds {
        let start = self.truncate_ms(t);
        let stop = self.period.add_ms(start);

        Bounds::new_checked(start, stop)
    }

    pub(crate) fn estimate_overlapping_bounds_ns(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ns()
            + self.period.duration_ns() / self.every.duration_ns()) as usize
    }

    pub(crate) fn estimate_overlapping_bounds_ms(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration_ms()
            + self.period.duration_ms() / self.every.duration_ms()) as usize
    }

    pub fn get_overlapping_bounds_iter(&self, boundary: Bounds, tu: TimeUnit) -> BoundsIter {
        BoundsIter::new(*self, boundary, tu)
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
    fn new(window: Window, boundary: Bounds, tu: TimeUnit) -> Self {
        let bi = match tu {
            TimeUnit::Nanoseconds => window.get_earliest_bounds_ns(boundary.start),
            TimeUnit::Milliseconds => window.get_earliest_bounds_ms(boundary.start),
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
