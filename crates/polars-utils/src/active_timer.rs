use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Indicates if the timer is currently ticking.
const TICKING_BIT: u64 = 1 << 63;

/// Measures the time for which at least 1 session was active.
#[derive(Debug)]
pub struct LiveTimer {
    base_instant: Instant,
    num_active: AtomicU64,
    /// If `TICKING_BIT` is set, this represents the amount of nanoseconds in `base_instant.elapsed()`
    /// for which the timer was not ticking. Otherwise, this represents the total amount of
    /// nanoseconds for which this timer was ticking.
    state_ns: AtomicU64,
    /// Used by `total_active_time_ns` to ensure output is monotonically nondecreasing.
    max_measured_ns: AtomicU64,
}

impl Default for LiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTimer {
    pub fn new() -> Self {
        Self {
            base_instant: Instant::now(),
            num_active: AtomicU64::new(0),
            state_ns: AtomicU64::new(0),
            max_measured_ns: AtomicU64::new(0),
        }
    }

    /// Register a session. This starts the timer if it is stopped.
    ///
    /// # Returns
    /// Returns a session guard that unregisters the session when dropped. The timer is stopped if
    /// no sessions are active after this guard is dropped.
    pub fn start_session(&self) -> LiveTimerSessionGuard<'_> {
        self._start_session();
        LiveTimerSessionGuard::new(self)
    }

    pub fn start_session_owned(self: Arc<Self>) -> LiveTimerSessionGuardOwned {
        self._start_session();
        LiveTimerSessionGuardOwned::new(self)
    }

    pub fn _start_session(&self) {
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
        }
    }

    fn _stop_session(&self) {
        let mut state_ns = u64::MAX;

        let _ = self.num_active.fetch_update(
            // # Release
            // * If this thread stops the timer, releases the store on `state_ns` below for the next
            //   thread that starts the timer.
            // * If this thread started the timer from `_start_session()`, but does not stop the
            //   timer, releases the store on `state_ns` from `_start_session()` for the thread
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

    pub fn total_active_time_ns(&self) -> u64 {
        let state_ns = self.state_ns.load(Ordering::Relaxed);

        let active_time = if state_ns & TICKING_BIT == 0 {
            state_ns
        } else {
            self.base_instant.elapsed().as_nanos() as u64 - (state_ns & !TICKING_BIT)
        };

        self.max_measured_ns
            .fetch_max(active_time, Ordering::Relaxed)
    }
}

pub struct LiveTimerSessionGuard<'a> {
    timer: &'a LiveTimer,
}

impl<'a> LiveTimerSessionGuard<'a> {
    fn new(timer: &'a LiveTimer) -> Self {
        Self { timer }
    }
}

impl Drop for LiveTimerSessionGuard<'_> {
    fn drop(&mut self) {
        self.timer._stop_session();
    }
}

pub struct LiveTimerSessionGuardOwned {
    timer: Arc<LiveTimer>,
}

impl LiveTimerSessionGuardOwned {
    fn new(timer: Arc<LiveTimer>) -> Self {
        Self { timer }
    }
}

impl Drop for LiveTimerSessionGuardOwned {
    fn drop(&mut self) {
        self.timer._stop_session();
    }
}

fn main() {
    use std::time::Duration;

    let timer = Arc::new(LiveTimer::new());

    let h1 = {
        let timer = timer.clone();
        std::thread::spawn(move || {
            timer._start_session();
            timer._stop_session();
            std::thread::sleep(Duration::from_secs(1));
            timer._start_session();
            std::thread::sleep(Duration::from_secs(1));
            timer._start_session();
            std::thread::sleep(Duration::from_secs(1));
            timer._stop_session();
            std::thread::sleep(Duration::from_secs(1));
            timer._stop_session();
            std::thread::sleep(Duration::from_secs(1));
        })
    };

    let mut recorded = [0u64; 16];

    'outer: while !h1.is_finished() {
        let mut prev = timer.total_active_time_ns();
        for i in recorded.iter_mut() {
            let new = timer.total_active_time_ns();

            if new < prev {
                break 'outer;
            }

            *i = new;
            prev = new;
        }
    }

    dbg!(&recorded);
    dbg!(recorded.is_sorted());

    dbg!(h1.join().unwrap());

    dbg!(timer.total_active_time_ns());
    dbg!(timer.max_measured_ns.load(Ordering::Relaxed));
}
