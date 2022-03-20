use super::groupby::ClosedWindow;

#[derive(Copy, Clone, Debug)]
pub struct Bounds {
    pub(crate) start: i64,
    pub(crate) stop: i64,
}

impl Bounds {
    /// Create a new `Bounds` and check the input is correct.
    pub(crate) fn new_checked(start: i64, stop: i64) -> Self {
        assert!(
            start <= stop,
            "boundary start must be smaller than stop; is your time column sorted in ascending order?\
            \nIf you did a groupby, note that null values are a separate group."
        );
        Self::new(start, stop)
    }

    /// Create a new `Bounds` without checking input correctness.
    pub(crate) fn new(start: i64, stop: i64) -> Self {
        Bounds { start, stop }
    }

    /// Duration in unit for this Boundary
    pub(crate) fn duration(&self) -> i64 {
        self.stop - self.start
    }

    // check if unit is within bounds
    pub(crate) fn is_member(&self, t: i64, closed: ClosedWindow) -> bool {
        match closed {
            ClosedWindow::Right => t > self.start && t <= self.stop,
            ClosedWindow::Left => t >= self.start && t < self.stop,
            ClosedWindow::None => t > self.start && t < self.stop,
            ClosedWindow::Both => t >= self.start && t <= self.stop,
        }
    }

    pub(crate) fn is_future(&self, t: i64, closed: ClosedWindow) -> bool {
        match closed {
            ClosedWindow::Left | ClosedWindow::None => self.stop <= t,
            ClosedWindow::Both | ClosedWindow::Right => t > self.stop,
        }
    }
}
