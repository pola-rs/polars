use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

/// Byte-granularity budget with dynamic resize.
///
/// Uses an atomic counter + Notify rather than tokio::Semaphore because
/// we need instant shrink semantics (tokio::Semaphore requires "stealing"
/// permits by acquiring them, which blocks under saturation).
#[derive(Debug)]
pub(super) struct ByteBudget {
    // Current budget, measured in number of bytes.
    current: AtomicU64,
    // Budget in use for in-flight traffic.
    inflight: AtomicU64,
    // Lowest allowed budget.
    floor: u64,
    waiters: Notify,
}

impl ByteBudget {
    fn new(initial: u64, floor: u64) -> Self {
        Self {
            current: AtomicU64::new(initial),
            floor,
            inflight: AtomicU64::new(0),
            waiters: Notify::new(),
        }
    }

    async fn acquire(&self, n_bytes: u64) {
        // Pre-empt deadlock.
        assert!(n_bytes <= self.floor);

        // NOTE: Large waiters can starve under sustained small-request load.
        // In practice, this may not be material issue.
        loop {
            let cap = self.current.load(Ordering::Acquire);
            let inflight = self.inflight.load(Ordering::Acquire);

            if inflight + n_bytes <= cap {
                if self
                    .inflight
                    .compare_exchange_weak(
                        inflight,
                        inflight + n_bytes,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // Fits. There may be leftover capacity for the next
                    // waiter (e.g. one big release satisfying several small
                    // acquires), so keep the wake chain alive — but only
                    // because progress occurred.
                    self.waiters.notify_one();
                    return;
                }
                continue;
            }

            // Doesn't fit: register, re-check, park.
            let notified = self.waiters.notified();
            let cap = self.current.load(Ordering::Acquire);
            let inflight = self.inflight.load(Ordering::Acquire);
            if inflight + n_bytes <= cap {
                continue;
            }
            notified.await;
        }
    }

    fn release(&self, bytes: u64) {
        self.inflight.fetch_sub(bytes, Ordering::AcqRel);
        self.waiters.notify_one();
    }

    fn resize(&self, new: u64) {
        let new = new.max(self.floor);
        let old = self.current.swap(new, Ordering::AcqRel);
        if new > old {
            // Grow: maybe someone can now proceed.
            self.waiters.notify_waiters();
        }
    }

    fn current_budget(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    fn floor_byte_budget(&self) -> u64 {
        self.floor
    }

    fn inflight(&self) -> u64 {
        self.inflight.load(Ordering::Relaxed)
    }
}

/// Request_count-based budget with fixed size.
#[derive(Debug)]
pub(super) struct RequestBudget {
    budget: usize,
    semaphore: Arc<Semaphore>,
    // // Budget in use for in-flight traffic.
    // inflight: AtomicU64,
}

impl RequestBudget {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            semaphore: Arc::new(Semaphore::new(budget)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InFlightStats {
    pub bytes_budget: u64,
    pub bytes_in_use: u64,
    // May exceed 1.0 transiently after a budget shrink, while
    // previously-admitted traffic drains. Expected, not a bug.
    pub bytes_saturation: f64,
    pub request_budget: usize,
    pub requests_in_use: usize,
    pub requests_saturation: f64,
}

#[derive(Debug)]
pub struct InFlightBudget {
    byte_budget: Arc<ByteBudget>,
    request_budget: Arc<RequestBudget>,
}

impl InFlightBudget {
    pub fn new(
        initial_byte_budget: u64,
        floor_byte_budget: u64,
        initial_request_budget: u32,
    ) -> Self {
        let inflight_budget = Self {
            byte_budget: Arc::new(ByteBudget::new(initial_byte_budget, floor_byte_budget)),
            request_budget: Arc::new(RequestBudget::new(initial_request_budget as usize)),
        };

        if polars_config::config().verbose() {
            eprintln!(
                "[InFlightConcurrency]: \
                initial_byte_budget: {}, \
                floor_byte_budget: {}, \
                request_budget: {}",
                initial_byte_budget, floor_byte_budget, initial_request_budget
            );
        }

        inflight_budget
    }

    pub async fn acquire(self: &Arc<Self>, n_bytes: u64) -> InFlightPermit {
        // kdn TODO INVESTIGATE: Note that merge_ranges and split_ranges may overshoot the floor.
        let n_bytes = n_bytes.min(self.byte_budget.floor_byte_budget());

        // Byte budget (may wait). Cancel-safe internally.
        self.byte_budget.acquire(n_bytes).await;

        // Guard immediately — synchronous, so there's no cancellation
        // window between reservation and guard.
        let bytes = ByteReservation {
            budget: self.byte_budget.clone(),
            n_bytes,
        };

        // Request permit. If we're cancelled here, `bytes` drops and
        // releases the reservation (and notifies a waiter).
        let req_permit = self
            .request_budget
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed");

        InFlightPermit {
            _byte_reservation: bytes,
            _req_permit: req_permit,
        }
    }

    pub fn current_byte_budget(&self) -> u64 {
        self.byte_budget.current_budget()
    }

    pub fn floor_byte_budget(&self) -> u64 {
        self.byte_budget.floor_byte_budget()
    }

    pub fn resize_byte_budget(&self, new: u64) {
        self.byte_budget.resize(new);
    }

    pub fn stats(&self) -> InFlightStats {
        let bytes_budget = self.byte_budget.current_budget();
        let bytes_in_use = self.byte_budget.inflight();
        let bytes_saturation = if bytes_budget > 0 {
            bytes_in_use as f64 / bytes_budget as f64
        } else {
            0.0
        };

        let request_budget = self.request_budget.budget;
        let requests_in_use = request_budget - self.request_budget.semaphore.available_permits();
        let requests_saturation = if request_budget > 0 {
            requests_in_use as f64 / request_budget as f64
        } else {
            0.0
        };

        InFlightStats {
            bytes_budget,
            bytes_in_use,
            bytes_saturation,
            request_budget,
            requests_in_use,
            requests_saturation,
        }
    }
}

struct ByteReservation {
    budget: Arc<ByteBudget>,
    n_bytes: u64,
}

impl Drop for ByteReservation {
    fn drop(&mut self) {
        self.budget.release(self.n_bytes);
    }
}

pub struct InFlightPermit {
    _byte_reservation: ByteReservation,
    _req_permit: OwnedSemaphorePermit,
}
