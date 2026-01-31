use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Measures the time for which at least 1 session was active.
#[derive(Debug)]
pub struct ActiveTimer {
    base_instant: Instant,
    num_active: AtomicU64,
    /// If `RUNNING_BIT` is set, this represents the amount of time to exclude from `base_instant.elapsed()`.
    /// Otherwise, this represents the total time recorded by this timer.
    state_ns: AtomicU64,
}

impl Default for ActiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

const RUNNING_BIT: u64 = 1 << 63;

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
        if self.num_active.fetch_add(1, Ordering::Acquire) == 0 {
            let mut state_ns = self.state_ns.load(Ordering::Relaxed);

            state_ns = self.base_instant.elapsed().as_nanos() as u64 - state_ns;

            self.state_ns
                .store(state_ns | RUNNING_BIT, Ordering::Relaxed);
        }
    }

    pub fn _unregister_session(&self) {
        let mut state_ns = u64::MAX;

        let _ = self
            .num_active
            .fetch_update(Ordering::Release, Ordering::Acquire, |num_active| {
                if num_active == 1 {
                    if state_ns & RUNNING_BIT != 0 {
                        if state_ns == u64::MAX {
                            state_ns = self.state_ns.load(Ordering::Relaxed);
                        }

                        state_ns &= !RUNNING_BIT;
                        state_ns = self.base_instant.elapsed().as_nanos() as u64 - state_ns;

                        self.state_ns.store(state_ns, Ordering::Relaxed);
                    }
                } else if state_ns & RUNNING_BIT == 0 {
                    state_ns = self.base_instant.elapsed().as_nanos() as u64 - state_ns;
                    state_ns |= RUNNING_BIT;
                    self.state_ns.store(state_ns, Ordering::Relaxed);
                }

                Some(num_active - 1)
            });
    }

    pub fn total_active_time_ns(&self) -> u64 {
        let state_ns = self.state_ns.load(Ordering::Relaxed);

        if state_ns & RUNNING_BIT == 0 {
            state_ns
        } else {
            self.base_instant.elapsed().as_nanos() as u64 - (state_ns & !RUNNING_BIT)
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
