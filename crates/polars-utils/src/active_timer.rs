use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Measures the time for which at least 1 session was active.
#[derive(Debug)]
pub struct ActiveTimer {
    base_instant: Instant,
    num_active: AtomicU64,
    start_ns: AtomicU64,
    total_ns: AtomicU64,
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
            start_ns: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
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
            self.start_ns.store(
                Instant::now().duration_since(self.base_instant).as_nanos() as u64,
                Ordering::Relaxed,
            );

            let total_ns = self.total_ns.load(Ordering::Relaxed);
            self.total_ns
                .store(total_ns | RUNNING_BIT, Ordering::Release);
        }
    }

    pub fn _unregister_session(&self) {
        let mut stopped = false;
        let mut stopped_at_ns = u64::MAX;
        let mut total_ns = u64::MAX;

        let _ = self
            .num_active
            .fetch_update(Ordering::Release, Ordering::Acquire, |num_active| {
                if num_active == 1 {
                    if !stopped {
                        let now_ns =
                            Instant::now().duration_since(self.base_instant).as_nanos() as u64;

                        if stopped_at_ns == u64::MAX {
                            stopped_at_ns = self.start_ns.load(Ordering::Relaxed);
                            total_ns = self.total_ns.load(Ordering::Relaxed);
                        }

                        let update = now_ns - stopped_at_ns;
                        total_ns += update;

                        self.total_ns
                            .store(total_ns & !RUNNING_BIT, Ordering::Relaxed);

                        stopped_at_ns = now_ns;

                        stopped = true;
                    }
                } else if stopped == true {
                    self.start_ns.store(
                        stopped_at_ns,
                        // Release the store on `total_ns` from the `!stopped` branch above.
                        Ordering::Release,
                    );
                    self.total_ns
                        .store(total_ns | RUNNING_BIT, Ordering::Release);

                    stopped = false;
                }

                Some(num_active - 1)
            });
    }

    pub fn total_active_time_ns(&self) -> u64 {
        let total_ns = self.total_ns.load(Ordering::Acquire);

        if total_ns & RUNNING_BIT == 0 {
            return total_ns;
        }

        let start_ns = self.start_ns.load(Ordering::Acquire);
        let total_ns2 = self.total_ns.load(Ordering::Relaxed);

        let total_ns = if total_ns2 != total_ns {
            // This is a freshly updated value
            total_ns2
        } else {
            let now_ns = Instant::now().duration_since(self.base_instant).as_nanos() as u64;
            let update = now_ns - start_ns;

            total_ns + update
        };

        total_ns & !RUNNING_BIT
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
