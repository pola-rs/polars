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
    // Lowest possible budget.
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

    async fn acquire(&self, bytes: u64) {
        // #kdn TODO PERF AI: this spins on notification; could starve large requests
        // if small ones keep grabbing released capacity. Consider a proper waker
        // queue with FIFO ordering or size-aware priority.

        // Pre-empt deadlock.
        assert!(bytes <= self.floor);
        loop {
            let cap = self.current.load(Ordering::Acquire);
            let inflight = self.inflight.load(Ordering::Acquire);

            if inflight + bytes <= cap {
                match self.inflight.compare_exchange_weak(
                    inflight,
                    inflight + bytes,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Update peak (racy, acceptable for a metric)
                        // self.peak_inflight.fetch_max(inflight + bytes, Ordering::Relaxed);
                        return;
                    },
                    Err(_) => continue,
                }
            } else {
                // Wait for a release event
                let notified = self.waiters.notified();
                // Re-check before awaiting (avoid missed wakeup)
                let cap = self.current.load(Ordering::Acquire);
                let inflight = self.inflight.load(Ordering::Acquire);
                if inflight + bytes <= cap {
                    continue;
                }
                notified.await;
            }
        }
    }

    fn release(&self, bytes: u64) {
        self.inflight.fetch_sub(bytes, Ordering::AcqRel);
        // #kdn TODO PERF AI: notify_waiters() causes thundering herd. Consider
        // notify_one() chained on release, but that requires tracking waiters
        // explicitly.
        self.waiters.notify_waiters();
    }

    fn resize(&self, new: u64) {
        let old = self.current.swap(new, Ordering::AcqRel);
        if new > old {
            // Grew: maybe someone can now proceed.
            self.waiters.notify_waiters();
        }
        // Shrunk: no need to notify. Existing in-flight complete normally;
        // new acquires see the smaller cap on next check.
    }

    fn current_budget(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    fn inflight(&self) -> u64 {
        self.inflight.load(Ordering::Relaxed)
    }
}

/// Request_count-based budget with fixed size.
#[derive(Debug)]
pub(super) struct RequestBudget {
    current: usize,
    semaphore: Arc<Semaphore>,
    // // Budget in use for in-flight traffic.
    // inflight: AtomicU64,
}

impl RequestBudget {
    fn new(budget: usize) -> Self {
        Self {
            current: budget,
            semaphore: Arc::new(Semaphore::new(
                usize::try_from(budget).expect("request budget exceeds usize limit"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InFlightStats {
    pub byte_budget: u64,
    pub bytes_in_use: u64,
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
            // current_request_budget: AtomicU32::new(initial_request_budget),
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

    pub async fn acquire(self: &Arc<Self>, bytes: u64) -> InFlightPermit {
        // Acquire request permit first (usually instant)
        let req_permit = self
            .request_budget
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed");

        // Then byte budget (may wait)
        self.byte_budget.acquire(bytes).await;

        InFlightPermit {
            admission: self.clone(),
            bytes,
            _req_permit: req_permit,
        }
    }

    pub fn current_byte_budget(&self) -> u64 {
        self.byte_budget.current_budget()
    }

    pub async fn resize_byte_budget(&self, new: u64) {
        self.byte_budget.resize(new);
    }

    pub fn stats(&self) -> InFlightStats {
        let byte_budget = self.byte_budget.current_budget();
        let bytes_in_use = self.byte_budget.inflight();
        let bytes_saturation = if byte_budget > 0 {
            bytes_in_use as f64 / byte_budget as f64
        } else {
            0.0
        };

        let request_budget = self.request_budget.current;
        let requests_in_use =
            request_budget as usize - self.request_budget.semaphore.available_permits();
        let requests_saturation = if request_budget > 0 {
            requests_in_use as f64 / request_budget as f64
        } else {
            0.0
        };

        InFlightStats {
            byte_budget,
            bytes_in_use,
            bytes_saturation,
            request_budget,
            requests_in_use,
            requests_saturation,
        }
    }
}

pub struct InFlightPermit {
    admission: Arc<InFlightBudget>,
    bytes: u64,
    _req_permit: OwnedSemaphorePermit,
}

impl InFlightPermit {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.admission.byte_budget.release(self.bytes);
    }
}
