use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Indicates if the timer is currently ticking.
const TICKING_BIT: u64 = 1 << 63;

/// Counts for how much time this timer was 'live'.
///
/// It is live when there is at least one session, but multiple concurrent
/// sessions don't increase the rate at which the timer ticks.
#[derive(Debug)]
pub struct LiveTimer {
    base_timestamp: Instant,
    live_sessions: AtomicU64,
    /// If `TICKING_BIT` is set, this represents the amount of nanoseconds in `base_timestamp.elapsed()`
    /// for which the timer was not ticking. Otherwise, this represents the total amount of
    /// nanoseconds for which this timer was ticking.
    state_ns: AtomicU64,
}

impl Default for LiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTimer {
    pub fn new() -> Self {
        Self {
            base_timestamp: Instant::now(),
            live_sessions: AtomicU64::new(0),
            state_ns: AtomicU64::new(0),
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
        if self.live_sessions.fetch_add(
            1,
            // Acquire the store on `state_ns` if the timer was previously stopped from another thread.
            Ordering::Acquire,
        ) == 0
        {
            let mut state_ns = self.state_ns.load(Ordering::Relaxed);

            state_ns = self.base_timestamp.elapsed().as_nanos() as u64 - state_ns;
            state_ns |= TICKING_BIT;

            self.state_ns.store(state_ns, Ordering::Relaxed);
        }
    }

    fn stop_session(&self) {
        let mut state_ns = u64::MAX;
        let mut stopped_at_ns = u64::MAX;

        let _ = self.live_sessions.fetch_update(
            // # Release
            // * If this thread stops the timer, releases the store on `state_ns` below for the next
            //   thread that starts the timer.
            // * If this thread started the timer from `_start_session()`, but does not stop the
            //   timer, releases the store on `state_ns` from `_start_session()` for the thread
            //   that eventually stops the timer.
            Ordering::Release,
            // Acquire the store on `state_ns` if the timer was started from another thread.
            Ordering::Acquire,
            |live_sessions| {
                if live_sessions == 1 {
                    if state_ns & TICKING_BIT != 0 {
                        if state_ns == u64::MAX {
                            state_ns = self.state_ns.load(Ordering::Relaxed);
                            stopped_at_ns = self.base_timestamp.elapsed().as_nanos() as u64;
                        }

                        state_ns = stopped_at_ns - (state_ns & !TICKING_BIT);

                        self.state_ns.store(state_ns, Ordering::Relaxed);
                    }
                } else if state_ns & TICKING_BIT == 0 {
                    // We stopped the timer, but another thread incremented `live_sessions`, so start
                    // the timer again.
                    state_ns = stopped_at_ns - state_ns;
                    state_ns |= TICKING_BIT;
                    self.state_ns.store(state_ns, Ordering::Relaxed);
                }

                Some(live_sessions - 1)
            },
        );
    }

    pub fn total_time_live_ns(&self) -> u64 {
        let state_ns = self.state_ns.load(Ordering::Relaxed);

        if state_ns & TICKING_BIT == 0 {
            return state_ns;
        }

        let curr_ns = self.base_timestamp.elapsed().as_nanos() as u64;

        // It could be that `stop_session()` performs a store on `state_ns` calculated against a
        // timestamp earlier than `curr_ns`, but this store was not yet visible to our earlier load
        // of `state_ns`. If we would report the elapsed time against `curr_ns` in this case, out
        // output on a subsequent call would report a decreased time.
        //
        // To handle this case, we perform the following spin in the hopes that it stalls for enough
        // time such that the store on `state_ns` from `stop_session()` becomes visible to us.
        for _ in 0..8 {
            let state_ns = self.state_ns.load(Ordering::Relaxed);

            if state_ns & TICKING_BIT == 0 {
                return state_ns;
            }

            std::hint::spin_loop();
        }

        curr_ns - (state_ns & !TICKING_BIT)
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
        self.timer.stop_session();
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
        self.timer.stop_session();
    }
}

fn main() {
    use std::sync::Arc;
    use std::time::Duration;

    let timer = Arc::new(LiveTimer::new());

    let h1 = {
        let timer = timer.clone();
        std::thread::spawn(move || {
            std::mem::forget(timer.start_session());
            timer.stop_session();
            dbg!(timer.total_time_live_ns());

            std::thread::sleep(Duration::from_secs(1));
            dbg!(timer.total_time_live_ns());

            std::mem::forget(timer.start_session());
            std::thread::sleep(Duration::from_secs(1));
            dbg!(timer.total_time_live_ns());

            std::mem::forget(timer.start_session());
            std::thread::sleep(Duration::from_secs(1));
            dbg!(timer.total_time_live_ns());

            timer.stop_session();
            std::thread::sleep(Duration::from_secs(1));
            dbg!(timer.total_time_live_ns());

            timer.stop_session();
            std::thread::sleep(Duration::from_secs(1));
            dbg!(timer.total_time_live_ns());
        })
    };

    let mut recorded = [0u64; 16];

    'outer: while !h1.is_finished() {
        let mut prev = 0;
        for i in recorded.iter_mut() {
            let new = timer.total_time_live_ns();

            *i = new;

            if new < prev {
                break 'outer;
            }

            prev = new;
        }
    }

    dbg!(&recorded);
    dbg!(recorded.is_sorted());

    dbg!(h1.join().unwrap());

    dbg!(timer.total_time_live_ns());
    dbg!(timer.state_ns.load(Ordering::Relaxed));
}
