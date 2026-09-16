use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Notify;

/// Byte-granularity budget with dynamic resize.
///
/// Uses an atomic counter + Notify rather than tokio::Semaphore because
/// we need instant shrink semantics (tokio::Semaphore requires "stealing"
/// permits by acquiring them, which blocks under saturation).
#[derive(Debug)]
pub(super) struct ByteBudget {
    // Current budget, measured in number of bytes, reflecting the maximum
    // allowed in-flight volume.
    current_budget: AtomicU64,
    // Lowest allowed budget for the current_budget.
    floor_budget: u64,
    // Volume in use for in-flight traffic, as allowed by the current_budget.
    inflight_in_use: AtomicU64,
    waiters: Notify,
}

impl ByteBudget {
    fn new(initial: u64, floor_budget: u64) -> Self {
        Self {
            current_budget: AtomicU64::new(initial),
            floor_budget,
            inflight_in_use: AtomicU64::new(0),
            waiters: Notify::new(),
        }
    }

    /// Acquire a bytes-based permit. The call site is responsible for capping the
    /// request size to prevent deadlock.
    async fn acquire_strict(&self, n_bytes: u64) {
        // Pre-empt deadlock.
        assert!(n_bytes <= self.floor_budget);

        // NOTE: Large waiters can starve under sustained small-request load.
        // In practice, this may not be material issue.
        loop {
            let cap = self.current_budget.load(Ordering::Acquire);
            let inflight = self.inflight_in_use.load(Ordering::Acquire);

            if inflight + n_bytes <= cap {
                if self
                    .inflight_in_use
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
            let cap = self.current_budget.load(Ordering::Acquire);
            let inflight = self.inflight_in_use.load(Ordering::Acquire);
            if inflight + n_bytes <= cap {
                continue;
            }
            notified.await;
        }
    }

    fn release(&self, bytes: u64) {
        self.inflight_in_use.fetch_sub(bytes, Ordering::AcqRel);
        self.waiters.notify_one();
    }

    fn resize(&self, new: u64) {
        let new = new.max(self.floor_budget);
        let old = self.current_budget.swap(new, Ordering::AcqRel);
        if new > old {
            // Grow: maybe someone can now proceed.
            self.waiters.notify_waiters();
        }
    }

    fn current_budget(&self) -> u64 {
        self.current_budget.load(Ordering::Relaxed)
    }

    fn floor_byte_budget(&self) -> u64 {
        self.floor_budget
    }

    fn inflight_in_use(&self) -> u64 {
        self.inflight_in_use.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub(super) struct RequestBudget {
    // Current budget, measured in count, reflecting the maximum allowed in-flight.
    current_budget: AtomicU64,
    // Lowest allowed budget for the current_budget.
    floor_budget: u64,
    // Count in use for in-flight traffic, as allowed by the current_budget.
    inflight_in_use: AtomicU64,
    waiters: Notify,
}

impl RequestBudget {
    fn new(initial: u64, floor_budget: u64) -> Self {
        Self {
            current_budget: AtomicU64::new(initial),
            floor_budget,
            inflight_in_use: AtomicU64::new(0),
            waiters: Notify::new(),
        }
    }

    /// Acquire a count-based permit. The call site is responsible for capping the
    /// request size to prevent deadlock.
    async fn acquire_one(&self) {
        let n = 1;

        loop {
            let cap = self.current_budget.load(Ordering::Acquire);
            let inflight = self.inflight_in_use.load(Ordering::Acquire);

            if inflight + n <= cap {
                if self
                    .inflight_in_use
                    .compare_exchange_weak(
                        inflight,
                        inflight + n,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // Fits.
                    self.waiters.notify_one();
                    return;
                }
                continue;
            }

            // Doesn't fit: register, re-check, park.
            let notified = self.waiters.notified();
            let cap = self.current_budget.load(Ordering::Acquire);
            let inflight = self.inflight_in_use.load(Ordering::Acquire);
            if inflight + n <= cap {
                continue;
            }
            notified.await;
        }
    }

    fn release_one(&self) {
        let n = 1;

        self.inflight_in_use.fetch_sub(n, Ordering::AcqRel);
        self.waiters.notify_one();
    }

    fn resize(&self, new: u64) {
        let new = new.max(self.floor_budget);
        let old = self.current_budget.swap(new, Ordering::AcqRel);
        if new > old {
            // Grow: maybe someone can now proceed.
            self.waiters.notify_waiters();
        }
    }

    fn inflight_in_use(&self) -> u64 {
        self.inflight_in_use.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InFlightStats {
    pub bytes_budget: u64,
    pub bytes_in_use: u64,
    // May exceed 1.0 transiently after a budget shrink, while
    // previously-admitted traffic drains. Expected, not a bug.
    pub bytes_saturation: f64,
    pub request_budget: u64,
    pub requests_in_use: u64,
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
        initial_request_budget: u64,
        floor_request_budget: u64,
    ) -> Self {
        let inflight_budget = Self {
            byte_budget: Arc::new(ByteBudget::new(initial_byte_budget, floor_byte_budget)),
            request_budget: Arc::new(RequestBudget::new(
                initial_request_budget,
                floor_request_budget,
            )),
        };

        if polars_config::config().verbose() {
            eprintln!(
                "[InFlightConcurrency]: \
                initial_byte_budget: {}, \
                floor_byte_budget: {}, \
                request_budget: {}, \
                floor_request_budget: {}",
                initial_byte_budget,
                floor_byte_budget,
                initial_request_budget,
                floor_request_budget
            );
        }

        inflight_budget
    }

    pub async fn acquire(self: &Arc<Self>, n_bytes: u64) -> InFlightPermit {
        // NOTE: since chunk_size is a target, merge_ranges and split_ranges may overshoot
        // the floor. Cap'ing prevents deadlock at the expense of memory management precision.
        let n_bytes = n_bytes.min(self.byte_budget.floor_byte_budget());

        // Byte budget (may wait). Cancel-safe internally.
        self.byte_budget.acquire_strict(n_bytes).await;

        // Guard immediately — synchronous, so there's no cancellation
        // window between reservation and guard.
        let bytes = BytesReservation {
            budget: self.byte_budget.clone(),
            n_bytes,
        };

        self.request_budget.acquire_one().await;

        let request = RequestReservation {
            budget: self.request_budget.clone(),
        };

        InFlightPermit {
            _bytes_reservation: bytes,
            _req_permit: request,
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

    pub fn resize_request_budget(&self, new: u64) {
        self.request_budget.resize(new);
    }

    pub fn stats(&self) -> InFlightStats {
        let bytes_budget = self.byte_budget.current_budget();
        let bytes_in_use = self.byte_budget.inflight_in_use();
        let bytes_saturation = if bytes_budget > 0 {
            bytes_in_use as f64 / bytes_budget as f64
        } else {
            0.0
        };

        let request_budget = self.request_budget.current_budget.load(Relaxed);
        let requests_in_use = self.request_budget.inflight_in_use();
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

struct BytesReservation {
    budget: Arc<ByteBudget>,
    n_bytes: u64,
}

impl Drop for BytesReservation {
    fn drop(&mut self) {
        self.budget.release(self.n_bytes);
    }
}

struct RequestReservation {
    budget: Arc<RequestBudget>,
}

impl Drop for RequestReservation {
    fn drop(&mut self) {
        self.budget.release_one();
    }
}

pub struct InFlightPermit {
    _bytes_reservation: BytesReservation,
    _req_permit: RequestReservation,
}
