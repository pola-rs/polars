use crate::calendar::timestamp_ns_to_datetime;
use crate::unit::TimeNanoseconds;
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone, Debug)]
pub struct Bounds {
    pub(crate) start: TimeNanoseconds,
    pub(crate) stop: TimeNanoseconds,
}

impl Display for Bounds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let start = timestamp_ns_to_datetime(*self.start);
        let end = timestamp_ns_to_datetime(*self.stop);
        write!(f, "Bounds: {} -> {}", start, end)
    }
}

// Get a wrapping boundary form a slice of nanoseconds.
// E.g. the first and last value are the start and stop of the boundary
impl<S: AsRef<[i64]>> From<S> for Bounds {
    fn from(s: S) -> Self {
        let slice = s.as_ref();
        let start = slice[0];
        let stop = slice[slice.len() - 1];
        Self::new(start.into(), stop.into())
    }
}

impl Bounds {
    pub fn new(start: TimeNanoseconds, stop: TimeNanoseconds) -> Self {
        Bounds { start, stop }
    }

    /// Duration in nanoseconcds for this Boundary
    pub fn duration(&self) -> TimeNanoseconds {
        (*self.stop - *self.start).into()
    }

    pub fn is_empty(&self) -> bool {
        *self.stop == *self.start
    }
}
