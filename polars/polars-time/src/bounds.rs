use crate::calendar::timestamp_ns_to_datetime;
use crate::groupby::ClosedWindow;
use crate::unit::TimeNanoseconds;
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone, Debug)]
pub struct Bounds {
    pub(crate) start: TimeNanoseconds,
    pub(crate) stop: TimeNanoseconds,
}

impl Display for Bounds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let start = timestamp_ns_to_datetime(self.start);
        let stop = timestamp_ns_to_datetime(self.stop);
        write!(f, "Bounds: {} -> {}", start, stop)
    }
}

// Get a wrapping boundary form a slice of nanoseconds.
// E.g. the first and last value are the start and stop of the boundary
impl<S: AsRef<[i64]>> From<S> for Bounds {
    fn from(s: S) -> Self {
        let slice = s.as_ref();
        let start = slice[0];
        let stop = slice[slice.len() - 1];
        Self::new(start, stop)
    }
}

impl Bounds {
    pub fn new(start: TimeNanoseconds, stop: TimeNanoseconds) -> Self {
        assert!(
            start < stop,
            "boundary start must be smaller than stop; is your time column sorted in ascending order?"
        );
        Bounds { start, stop }
    }

    /// Duration in nanoseconcds for this Boundary
    pub fn duration(&self) -> TimeNanoseconds {
        self.stop - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.stop == self.start
    }

    // check if nanoseconds is within bounds
    pub fn is_member(&self, t: i64, closed: ClosedWindow) -> bool {
        match closed {
            ClosedWindow::Right => t > self.start && t <= self.stop,
            ClosedWindow::Left => t >= self.start && t < self.stop,
            ClosedWindow::None => t > self.start && t < self.stop,
            ClosedWindow::Both => t >= self.start && t <= self.stop,
        }
    }

    pub fn is_future(&self, t: i64) -> bool {
        t > self.stop
    }
}
