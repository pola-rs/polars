use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const TICKING_BIT: u64 = 1 << 63; // Indicates if the timer is currently ticking.
const SESSION_UNIT: u64 = 1; // Base unit for number of sessions.
const SESSION_MASK: u64 = TIMER_UNIT - 1; // Mask used to extract session count.
const TIMER_UNIT: u64 = 1 << 32; // Base unit for number of timers.

/// Counts for how much time this timer was 'live'.
///
/// It is live when there is at least one session, but multiple concurrent
/// sessions don't increase the rate at which the timer ticks.
///
/// Clones of this timer are cheap, and the clones are identical, like an Arc.
pub struct LiveTimer {
    // We use a raw pointer instead of an Arc to ensure starting/stopping sessions only involves
    // a single atomic operation, while still letting LiveTimerSession be lifetime-free.
    inner: *mut LiveTimerInner,
}

unsafe impl Send for LiveTimer {}
unsafe impl Sync for LiveTimer {}

impl Default for LiveTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTimer {
    pub fn new() -> Self {
        let inner = LiveTimerInner {
            base_timestamp: Instant::now(),
            refcount: AtomicU64::new(TIMER_UNIT),
            _padding: [0; _],
            state_ns: AtomicU64::new(0),
            max_live_ns: AtomicU64::new(0),
        };

        Self {
            inner: Box::into_raw(Box::new(inner)),
        }
    }

    pub fn start_session(&self) -> LiveTimerSession {
        unsafe { (&*self.inner).start_session() };
        LiveTimerSession { inner: self.inner }
    }

    pub fn total_time_live_ns(&self) -> u64 {
        unsafe { (&*self.inner).total_time_live_ns() }
    }
}

impl Clone for LiveTimer {
    fn clone(&self) -> Self {
        let inner = unsafe { &*self.inner };
        inner.refcount.fetch_add(TIMER_UNIT, Ordering::Relaxed);
        Self { inner: self.inner }
    }
}

impl Drop for LiveTimer {
    fn drop(&mut self) {
        unsafe {
            let old_rc = (&*self.inner)
                .refcount
                .fetch_sub(TIMER_UNIT, Ordering::AcqRel);
            if old_rc == TIMER_UNIT {
                drop(Box::from_raw(self.inner))
            }
        }
    }
}

pub struct LiveTimerSession {
    inner: *mut LiveTimerInner,
}

unsafe impl Send for LiveTimerSession {}
unsafe impl Sync for LiveTimerSession {}

impl Clone for LiveTimerSession {
    fn clone(&self) -> Self {
        unsafe { (*self.inner).start_session() };
        Self { inner: self.inner }
    }
}

impl Drop for LiveTimerSession {
    fn drop(&mut self) {
        unsafe {
            if (*self.inner).stop_session() {
                drop(Box::from_raw(self.inner))
            }
        }
    }
}

struct LiveTimerInner {
    base_timestamp: Instant,
    /// Contains two 32-bit refcounts: number of timer references and number of live sessions.
    /// If both are zero this object is destroyed.
    refcount: AtomicU64,
    /// Ensures refcount (commonly modified) is on a different cache line to state_ns.
    _padding: [u8; 64],
    /// Contains the total amount of time the timer was live. Interpreted differently depending on TICKING_BIT:
    ///   0 -> Duration::from_nanos(state)
    ///   1 -> base_timestamp.elapsed() - Duration::from_nanos(state & !TICKING_BIT)
    state_ns: AtomicU64,
    /// Used by `total_time_live_ns` to ensure it returns monotonically nondecreasing values.
    max_live_ns: AtomicU64,
}

impl LiveTimerInner {
    fn start_session(&self) {
        // (1) Acquire to ensure we load state_ns from previous stop_session, if necessary.
        let prev_sessions = self.refcount.fetch_add(SESSION_UNIT, Ordering::Acquire) & SESSION_MASK;
        if prev_sessions == 0 {
            let orig_state_ns = self.state_ns.load(Ordering::Relaxed);
            let start_ns = self.base_timestamp.elapsed().as_nanos() as u64;
            debug_assert!(orig_state_ns & TICKING_BIT == 0);
            let new_state_ns = start_ns.saturating_sub(orig_state_ns) | TICKING_BIT;
            self.state_ns.store(new_state_ns, Ordering::Release); // See (2).
        }
    }

    /// Returns true if this timer should be destroyed.
    fn stop_session(&self) -> bool {
        let mut stopped = false;
        let mut stopped_at_ns = u64::MAX;
        let mut orig_state_ns = u64::MAX;

        // (1) Acquire and Release to ensure we load state_ns from previous start_session, if necessary.
        self.refcount
            .fetch_update(Ordering::Release, Ordering::Acquire, |rc| {
                if rc == SESSION_UNIT {
                    return None; // We're the sole reference, can just destroy.
                }

                // Stop or re-start the timer if necessary.
                let should_stop = rc & SESSION_MASK == SESSION_UNIT;
                if should_stop && !stopped {
                    if stopped_at_ns == u64::MAX {
                        orig_state_ns = self.state_ns.load(Ordering::Relaxed);
                        stopped_at_ns = self.base_timestamp.elapsed().as_nanos() as u64;
                    }
                    let new_state_ns = stopped_at_ns.saturating_sub(orig_state_ns & !TICKING_BIT);
                    self.state_ns.store(new_state_ns, Ordering::Relaxed);
                    stopped = true;
                } else if !should_stop && stopped {
                    self.state_ns.store(orig_state_ns, Ordering::Release); // See (2).
                    stopped = false;
                }

                Some(rc - SESSION_UNIT)
            })
            .is_err()
    }

    fn total_time_live_ns(&self) -> u64 {
        // (2) Acquire ensures the elapsed() call by this function is sequenced after the elapsed()
        // call that state_ns value was calculated against if it is currently ticking.
        let state_ns = self.state_ns.load(Ordering::Acquire);
        let active_time = if state_ns & TICKING_BIT == 0 {
            state_ns
        } else {
            let now_ns = self.base_timestamp.elapsed().as_nanos() as u64;
            now_ns.saturating_sub(state_ns & !TICKING_BIT)
        };

        // Needed to ensure monotonicity, our load above interleaved with stops/restarts
        // could otherwise be non-monotonic.
        u64::max(
            active_time,
            self.max_live_ns.fetch_max(active_time, Ordering::Relaxed),
        )
    }
}

impl std::fmt::Debug for LiveTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let refcount = unsafe { (&*self.inner).refcount.load(Ordering::Relaxed) };
        let active_sessions = refcount & SESSION_MASK;
        let total_time_live_ns = self.total_time_live_ns();

        return std::fmt::Debug::fmt(
            &display::LiveTimer {
                active_sessions,
                total_time_live_ns,
            },
            f,
        );

        mod display {
            #[derive(Debug)]
            #[expect(unused)]
            pub struct LiveTimer {
                pub active_sessions: u64,
                pub total_time_live_ns: u64,
            }
        }
    }
}

impl std::fmt::Debug for LiveTimerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return std::fmt::Debug::fmt(&display::LiveTimerSession {}, f);

        mod display {
            #[derive(Debug)]
            pub struct LiveTimerSession {}
        }
    }
}
