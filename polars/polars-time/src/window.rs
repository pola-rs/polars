use crate::bounds::Bounds;
use crate::duration::Duration;
use crate::unit::TimeNanoseconds;

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
    pub fn truncate(&self, t: TimeNanoseconds) -> TimeNanoseconds {
        self.every.truncate_nanoseconds(t) + self.offset
    }

    /// returns the bounds for the earliest window bounds
    /// that contains the given time t.  For underlapping windows that
    /// do not contain time t, the window directly after time t will be returned.
    pub fn get_earliest_bounds(&self, t: TimeNanoseconds) -> Bounds {
        // original code translates offset here
        // we don't. Seems unintuitive to me.
        let start = self.truncate(t);
        let stop = self.truncate(t) + self.period;

        Bounds::new(start, stop)
    }

    pub(crate) fn estimate_overlapping_bounds(&self, boundary: Bounds) -> usize {
        (boundary.duration() / self.every.duration()
            + self.period.duration() / self.every.duration()) as usize
    }

    pub fn get_overlapping_bounds(&self, boundary: Bounds) -> Vec<Bounds> {
        if boundary.is_empty() {
            return vec![];
        } else {
            // estimate size
            let size = self.estimate_overlapping_bounds(boundary);
            let mut out_bounds = Vec::with_capacity(size);

            for bi in self.get_overlapping_bounds_iter(boundary) {
                out_bounds.push(bi);
            }
            out_bounds
        }
    }

    pub fn get_overlapping_bounds_iter(&self, boundary: Bounds) -> BoundsIter {
        BoundsIter::new(*self, boundary)
    }
}

pub struct BoundsIter {
    window: Window,
    // wrapping boundary
    boundary: Bounds,
    // boundary per window iterator
    bi: Bounds,
}
impl BoundsIter {
    fn new(window: Window, boundary: Bounds) -> Self {
        let bi = window.get_earliest_bounds(boundary.start);
        Self {
            window,
            boundary,
            bi,
        }
    }
}

impl Iterator for BoundsIter {
    type Item = Bounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bi.start < self.boundary.stop {
            let out = self.bi;
            self.bi.start = self.bi.start + self.window.every;
            self.bi.stop = self.bi.stop + self.window.every;
            Some(out)
        } else {
            None
        }
    }
}
