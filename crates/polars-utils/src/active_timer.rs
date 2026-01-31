use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Measures the time for which at least 1 session was active.
#[derive(Debug)]
pub struct ActiveTimer {
    base_instant: Instant,
    num_active: AtomicU64,
    /// If `TICKING_BIT` is set, this represents the amount of nanoseconds in `base_instant.elapsed()`
    /// for which the timer was was not active. Otherwise, this represents the total amount of
    /// nanoseconds for which this timer was ticking.
    state_ns: AtomicU64,
}

impl Default for ActiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Indicates if this timer is currently ticking.
const TICKING_BIT: u64 = 1 << 63;

impl ActiveTimer {
    pub fn new() -> Self {
        Self {
            base_instant: Instant::now(),
            num_active: AtomicU64::new(0),
            state_ns: AtomicU64::new(0),
        }
    }

    /// Register a session. This starts the timer if it is stopped.
    ///
    /// # Returns
    /// Returns a session guard that unregisters the session when dropped. The timer is stopped if
    /// no sessions are active after this guard is dropped.
    pub fn register_session(&self) -> ActiveTimerSessionGuard<'_> {
        self._register_session();
        ActiveTimerSessionGuard::new(self)
    }

    pub fn register_session_owned(self: Arc<Self>) -> ActiveTimerSessionGuardOwned {
        self._register_session();
        ActiveTimerSessionGuardOwned::new(self)
    }

    pub fn _register_session(&self) {
        if self.num_active.fetch_add(
            1,
            // Acquire the store on `state_ns` if the timer was previously stopped from another thread.
            Ordering::Acquire,
        ) == 0
        {
            let mut state_ns = self.state_ns.load(Ordering::Relaxed);

            state_ns = self.base_instant.elapsed().as_nanos() as u64 - state_ns;

            self.state_ns
                .store(state_ns | TICKING_BIT, Ordering::Relaxed);
            // Release immediately, otherwise `total_active_time_ns()` may not see the started timer
            // until this thread reaches and completes `self._unregister_session()`.
            self.num_active.fetch_add(0, Ordering::Release);
        }
    }

    pub fn _unregister_session(&self) {
        let mut state_ns = u64::MAX;

        let _ = self.num_active.fetch_update(
            // # Release
            // * If this thread stops the timer, releases the store on `state_ns` below for the next
            //   thread that starts the timer.
            // * If this thread started the timer from `_register_session()`, but does not stop the
            //   timer, releases the store on `state_ns` from `_register_session()` for the thread
            //   that eventually stops the timer.
            Ordering::Release,
            // Acquire the store on `state_ns` if the timer was started from another thread.
            Ordering::Acquire,
            |num_active| {
                if num_active == 1 {
                    if state_ns & TICKING_BIT != 0 {
                        if state_ns == u64::MAX {
                            state_ns = self.state_ns.load(Ordering::Relaxed);
                        }

                        state_ns = self.base_instant.elapsed().as_nanos() as u64
                            - (state_ns & !TICKING_BIT);

                        self.state_ns.store(state_ns, Ordering::Relaxed);
                    }
                } else if state_ns & TICKING_BIT == 0 {
                    // We stopped the timer, but another thread incremented `num_active`, so start
                    // the timer again.
                    state_ns = self.base_instant.elapsed().as_nanos() as u64 - state_ns;
                    state_ns |= TICKING_BIT;
                    self.state_ns.store(state_ns, Ordering::Relaxed);
                }

                Some(num_active - 1)
            },
        );
    }

    /// Note, this is NOT monotonically nondecreasing across calls.
    pub fn total_active_time_ns(&self) -> u64 {
        self.num_active.load(Ordering::Acquire);
        let state_ns = self.state_ns.load(Ordering::Relaxed);

        if state_ns & TICKING_BIT == 0 {
            state_ns
        } else {
            self.base_instant.elapsed().as_nanos() as u64 - (state_ns & !TICKING_BIT)
        }
    }
}

pub use session::{ActiveTimerSessionGuard, ActiveTimerSessionGuardOwned};

mod session {
    use std::sync::Arc;

    use super::ActiveTimer;

    pub struct ActiveTimerSessionGuard<'a> {
        timer: &'a ActiveTimer,
    }

    impl<'a> ActiveTimerSessionGuard<'a> {
        pub(super) fn new(timer: &'a ActiveTimer) -> Self {
            Self { timer }
        }
    }

    impl Drop for ActiveTimerSessionGuard<'_> {
        fn drop(&mut self) {
            self.timer._unregister_session();
        }
    }

    pub struct ActiveTimerSessionGuardOwned {
        timer: Arc<ActiveTimer>,
    }

    impl ActiveTimerSessionGuardOwned {
        pub(super) fn new(timer: Arc<ActiveTimer>) -> Self {
            Self { timer }
        }
    }

    impl Drop for ActiveTimerSessionGuardOwned {
        fn drop(&mut self) {
            self.timer._unregister_session();
        }
    }
}
