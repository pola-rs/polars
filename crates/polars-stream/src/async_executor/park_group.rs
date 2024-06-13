use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// A group of workers that can park / unpark each other.
///
/// There is at most one worker at a time which is considered a 'recruiter'.
/// A recruiter hasn't yet found work and will either park again or recruit the
/// next worker when it finds work.
///
/// Calls to park/unpark participate in a global SeqCst order.
#[derive(Default)]
pub struct ParkGroup {
    inner: Arc<ParkGroupInner>,
}

#[derive(Default)]
struct ParkGroupInner {
    // The condvar we park with.
    condvar: Condvar,

    // Contains the number of notifications and whether or not the next unparked
    // worker should become a recruiter.
    notifications: Mutex<(u32, bool)>,

    // Bits  0..32: number of idle workers.
    // Bit      32: set if there is an active recruiter.
    // Bit      33: set if a worker is preparing to park.
    // Bits 34..64: version that is incremented to cancel a park request.
    state: AtomicU64,

    num_workers: AtomicU32,
}

const IDLE_UNIT: u64 = 1;
const ACTIVE_RECRUITER_BIT: u64 = 1 << 32;
const PREPARING_TO_PARK_BIT: u64 = 1 << 33;
const VERSION_UNIT: u64 = 1 << 34;

fn state_num_idle(state: u64) -> u32 {
    state as u32
}

fn state_version(state: u64) -> u32 {
    (state >> 34) as u32
}

pub struct ParkGroupWorker {
    inner: Arc<ParkGroupInner>,
    recruiter: bool,
    version: u32,
}

impl ParkGroup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new worker.
    ///
    /// # Panics
    /// Panics if you try to create more than 2^32 - 1 workers.
    pub fn new_worker(&self) -> ParkGroupWorker {
        self.inner
            .num_workers
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |w| w.checked_add(1))
            .expect("can't have more than 2^32 - 1 workers");

        ParkGroupWorker {
            version: 0,
            inner: Arc::clone(&self.inner),
            recruiter: false,
        }
    }

    /// Unparks an idle worker if there is no recruiter.
    ///
    /// Also cancels in-progress park attempts.
    pub fn unpark_one(&self) {
        self.inner.unpark_one();
    }
}

impl ParkGroupWorker {
    /// Prepares to park this worker.
    pub fn prepare_park(&mut self) -> ParkAttempt<'_> {
        let mut state = self.inner.state.load(Ordering::SeqCst);
        self.version = state_version(state);

        // If the version changes or someone else has set the
        // PREPARING_TO_PARK_BIT, stop trying to update the state.
        while state & PREPARING_TO_PARK_BIT == 0 && state_version(state) == self.version {
            // Notify that we're preparing to park, and while we're at it might as
            // well try to become a recruiter to avoid expensive unparks.
            let new_state = state | PREPARING_TO_PARK_BIT | ACTIVE_RECRUITER_BIT;
            match self.inner.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::Relaxed,
                Ordering::SeqCst,
            ) {
                Ok(s) => {
                    if s & ACTIVE_RECRUITER_BIT == 0 {
                        self.recruiter = true;
                    }
                    break;
                },

                Err(s) => state = s,
            }
        }

        ParkAttempt { worker: self }
    }

    /// You should call this function after finding work to recruit the next
    /// worker if this worker was a recruiter.
    pub fn recruit_next(&mut self) {
        if !self.recruiter {
            return;
        }

        // Recruit the next idle worker or mark that there is no recruiter anymore.
        let mut recruit_next = false;
        let _ = self
            .inner
            .state
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
                debug_assert!(state & ACTIVE_RECRUITER_BIT != 0);

                recruit_next = state_num_idle(state) > 0;
                let bit = if recruit_next {
                    IDLE_UNIT
                } else {
                    ACTIVE_RECRUITER_BIT
                };
                Some(state - bit)
            });

        if recruit_next {
            self.inner.unpark_one_slow_as_recruiter();
        }
        self.recruiter = false;
    }
}

pub struct ParkAttempt<'a> {
    worker: &'a mut ParkGroupWorker,
}

impl<'a> ParkAttempt<'a> {
    /// Actually park this worker.
    ///
    /// If there were calls to unpark between calling prepare_park() and park(),
    /// this park attempt is cancelled and immediately returns.
    pub fn park(mut self) {
        let state = &self.worker.inner.state;
        let update = state.fetch_update(Ordering::Relaxed, Ordering::SeqCst, |state| {
            if state_version(state) != self.worker.version {
                // We got notified of new work, cancel park.
                None
            } else if self.worker.recruiter {
                Some(state + IDLE_UNIT - ACTIVE_RECRUITER_BIT)
            } else {
                Some(state + IDLE_UNIT)
            }
        });

        if update.is_ok() {
            self.park_slow()
        }
    }

    #[cold]
    fn park_slow(&mut self) {
        let condvar = &self.worker.inner.condvar;
        let mut notifications = self.worker.inner.notifications.lock();
        condvar.wait_while(&mut notifications, |n| n.0 == 0);

        // Possibly become a recruiter and consume the notification.
        self.worker.recruiter = notifications.1;
        notifications.0 -= 1;
        notifications.1 = false;
    }
}

impl ParkGroupInner {
    fn unpark_one(&self) {
        let mut should_unpark = false;
        let _ = self
            .state
            .fetch_update(Ordering::Release, Ordering::SeqCst, |state| {
                should_unpark = state_num_idle(state) > 0 && state & ACTIVE_RECRUITER_BIT == 0;
                if should_unpark {
                    Some(state - IDLE_UNIT + ACTIVE_RECRUITER_BIT)
                } else if state & PREPARING_TO_PARK_BIT == PREPARING_TO_PARK_BIT {
                    Some(state.wrapping_add(VERSION_UNIT) & !PREPARING_TO_PARK_BIT)
                } else {
                    None
                }
            });

        if should_unpark {
            self.unpark_one_slow_as_recruiter();
        }
    }

    #[cold]
    fn unpark_one_slow_as_recruiter(&self) {
        let mut notifications = self.notifications.lock();
        notifications.0 += 1;
        notifications.1 = true;
        self.condvar.notify_one();
    }
}
