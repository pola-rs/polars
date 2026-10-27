use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, oneshot};

/// Order in which parked acquires of the in-flight byte budget are admitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmissionOrder {
    /// Lowest issue number first (`in_issue_order`); requests without one go first.
    Issue,
    /// Whichever waiter wins the race after a release; new arrivals may pass parked waiters.
    Arrival,
}

static NEXT_ISSUE_SEQ: AtomicU64 = AtomicU64::new(1);

tokio::task_local! {
    static ISSUE_SEQ: u64;
}

/// Runs `fut` with the next issue number, taken now, for its in-flight byte budget acquires. Call
/// it where fetches are issued in order, not where their tasks start.
pub fn in_issue_order<F: Future>(fut: F) -> impl Future<Output = F::Output> {
    ISSUE_SEQ.scope(NEXT_ISSUE_SEQ.fetch_add(1, Relaxed), fut)
}

/// Issue number of the current task, 0 if none.
fn current_issue_seq() -> u64 {
    ISSUE_SEQ.try_with(|seq| *seq).unwrap_or(0)
}

#[derive(Debug)]
struct Waiter {
    n_bytes: u64,
    granted: oneshot::Sender<()>,
}

/// Parked acquires keyed by (issue number, arrival number).
#[derive(Debug, Default)]
struct IssueQueue {
    next_arrival: u64,
    waiters: BTreeMap<(u64, u64), Waiter>,
}

/// Byte-granularity budget with dynamic resize.
///
/// Uses atomic counters with a Notify (arrival order) or a queue of grants (issue order) rather
/// than tokio::Semaphore because we need instant shrink semantics (tokio::Semaphore requires
/// "stealing" permits by acquiring them, which blocks under saturation).
#[derive(Debug)]
pub(super) struct ByteBudget {
    // Current budget, measured in number of bytes, reflecting the maximum
    // allowed in-flight volume.
    current_budget: AtomicU64,
    // Lowest allowed budget for the current_budget.
    floor_budget: u64,
    // Volume in use for in-flight traffic, as allowed by the current_budget.
    inflight_in_use: AtomicU64,
    // Gauge: acquires parked now.
    acquires_parked: AtomicU64,
    order: AdmissionOrder,
    // `AdmissionOrder::Arrival` waiters.
    waiters: Notify,
    // `AdmissionOrder::Issue` waiters, and their count (read without the lock).
    queue: Mutex<IssueQueue>,
    queued: AtomicU64,
}

impl ByteBudget {
    fn new(initial: u64, floor_budget: u64, order: AdmissionOrder) -> Self {
        Self {
            current_budget: AtomicU64::new(initial),
            floor_budget,
            inflight_in_use: AtomicU64::new(0),
            acquires_parked: AtomicU64::new(0),
            order,
            waiters: Notify::new(),
            queue: Mutex::default(),
            queued: AtomicU64::new(0),
        }
    }

    /// Acquire a bytes-based permit. The call site is responsible for capping the
    /// request size to prevent deadlock. Returns whether the caller parked.
    async fn acquire_strict(&self, n_bytes: u64, issue_seq: u64) -> bool {
        // Pre-empt deadlock.
        assert!(n_bytes <= self.floor_budget);

        if self.order == AdmissionOrder::Issue {
            return self.acquire_in_issue_order(n_bytes, issue_seq).await;
        }

        let mut parked = false;

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
                    return parked;
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
            let _parked = Parked::new(&self.acquires_parked);
            notified.await;
            parked = true;
        }
    }

    /// Waits behind every parked acquire with a lower issue number; passes parked acquires with an
    /// equal or higher issue number if the bytes fit. Returns whether it parked. The queue is only
    /// non-empty while its front does not fit, so every parked acquire waits on the budget.
    async fn acquire_in_issue_order(&self, n_bytes: u64, issue_seq: u64) -> bool {
        if self.queued.load(SeqCst) == 0 && self.try_take(n_bytes) {
            return false;
        }

        let (key, granted) = {
            let mut queue = self.queue.lock().unwrap();
            let lower_waits = queue
                .waiters
                .keys()
                .next()
                .is_some_and(|&(seq, _)| seq < issue_seq);
            if !lower_waits && self.try_take(n_bytes) {
                return false;
            }
            let key = (issue_seq, queue.next_arrival);
            queue.next_arrival += 1;
            let (tx, rx) = oneshot::channel();
            queue.waiters.insert(
                key,
                Waiter {
                    n_bytes,
                    granted: tx,
                },
            );
            // Counted before the grant below re-reads the bytes in use: a release that read
            // `queued == 0` freed its bytes before this re-read.
            self.queued.fetch_add(1, SeqCst);
            self.grant_locked(&mut queue);
            if !queue.waiters.contains_key(&key) {
                // Granted by the call above: the bytes are taken.
                return false;
            }
            (key, rx)
        };

        let mut wait = QueuedAcquire {
            budget: self,
            key,
            n_bytes,
            granted: false,
        };
        let _parked = Parked::new(&self.acquires_parked);
        // The sender is only dropped unsent when `wait` removes its own entry.
        let _ = granted.await;
        wait.granted = true;
        true
    }

    /// Takes `n_bytes` if they fit.
    fn try_take(&self, n_bytes: u64) -> bool {
        let cap = self.current_budget.load(SeqCst);
        let mut inflight = self.inflight_in_use.load(SeqCst);
        while inflight + n_bytes <= cap {
            match self.inflight_in_use.compare_exchange_weak(
                inflight,
                inflight + n_bytes,
                SeqCst,
                SeqCst,
            ) {
                Ok(_) => return true,
                Err(current) => inflight = current,
            }
        }
        false
    }

    /// Admits parked acquires from the front of the queue while they fit.
    fn grant_locked(&self, queue: &mut IssueQueue) {
        while let Some(front) = queue.waiters.first_entry() {
            if !self.try_take(front.get().n_bytes) {
                break;
            }
            self.queued.fetch_sub(1, SeqCst);
            // A cancelled acquire removes its entry under the lock, so the receiver is alive.
            let _ = front.remove().granted.send(());
        }
    }

    fn grant_if_queued(&self) {
        if self.queued.load(SeqCst) > 0 {
            self.grant_locked(&mut self.queue.lock().unwrap());
        }
    }

    fn release(&self, bytes: u64) {
        self.inflight_in_use.fetch_sub(bytes, SeqCst);
        match self.order {
            AdmissionOrder::Issue => self.grant_if_queued(),
            AdmissionOrder::Arrival => self.waiters.notify_one(),
        }
    }

    fn resize(&self, new: u64) {
        let new = new.max(self.floor_budget);
        let old = self.current_budget.swap(new, SeqCst);
        if new > old {
            // Grow: maybe someone can now proceed.
            match self.order {
                AdmissionOrder::Issue => self.grant_if_queued(),
                AdmissionOrder::Arrival => self.waiters.notify_waiters(),
            }
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

/// A parked `AdmissionOrder::Issue` acquire. Dropped before its grant (cancelled), it leaves the
/// queue; dropped after a grant it did not observe, it returns the bytes.
struct QueuedAcquire<'a> {
    budget: &'a ByteBudget,
    key: (u64, u64),
    n_bytes: u64,
    granted: bool,
}

impl Drop for QueuedAcquire<'_> {
    fn drop(&mut self) {
        if self.granted {
            return;
        }
        let mut queue = self.budget.queue.lock().unwrap();
        if queue.waiters.remove(&self.key).is_some() {
            self.budget.queued.fetch_sub(1, SeqCst);
        } else {
            self.budget.inflight_in_use.fetch_sub(self.n_bytes, SeqCst);
        }
        // It may have been the front of the queue.
        self.budget.grant_locked(&mut queue);
    }
}

/// Counts one parked acquire while alive (also when the acquire is cancelled).
struct Parked<'a>(&'a AtomicU64);

impl<'a> Parked<'a> {
    fn new(n: &'a AtomicU64) -> Self {
        n.fetch_add(1, Relaxed);
        Self(n)
    }
}

impl Drop for Parked<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Relaxed);
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
    // Gauge: byte acquires parked at the time of `stats()`.
    pub byte_acquires_parked: u64,
    pub request_budget: u64,
    pub requests_in_use: u64,
    pub requests_saturation: f64,
}

#[derive(Debug)]
pub struct InFlightBudget {
    byte_budget: Arc<ByteBudget>,
    request_budget: Arc<RequestBudget>,
    // Flow: byte acquires admitted after parking, since the last `take_byte_acquires_unparked()`.
    byte_acquires_unparked: AtomicU64,
}

impl InFlightBudget {
    pub fn new(
        initial_byte_budget: u64,
        floor_byte_budget: u64,
        initial_request_budget: u64,
        floor_request_budget: u64,
        admission_order: AdmissionOrder,
    ) -> Self {
        Self {
            byte_budget: Arc::new(ByteBudget::new(
                initial_byte_budget,
                floor_byte_budget,
                admission_order,
            )),
            request_budget: Arc::new(RequestBudget::new(
                initial_request_budget,
                floor_request_budget,
            )),
            byte_acquires_unparked: AtomicU64::new(0),
        }
    }

    pub async fn acquire(self: &Arc<Self>, n_bytes: u64) -> InFlightPermit {
        // NOTE: since chunk_size is a target, merge_ranges and split_ranges may overshoot
        // the floor. Cap'ing prevents deadlock at the expense of memory management precision.
        let n_bytes = n_bytes.min(self.byte_budget.floor_byte_budget());

        // Byte budget (may wait). Cancel-safe internally.
        if self
            .byte_budget
            .acquire_strict(n_bytes, current_issue_seq())
            .await
        {
            self.byte_acquires_unparked.fetch_add(1, Relaxed);
        }

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

    /// Flow: byte acquires admitted after parking, since the last call. Read and reset.
    pub fn take_byte_acquires_unparked(&self) -> u64 {
        self.byte_acquires_unparked.swap(0, Relaxed)
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
            byte_acquires_parked: self.byte_budget.acquires_parked.load(Relaxed),
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

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    use super::*;

    fn poll<F: Future>(f: &mut Pin<Box<F>>) -> Poll<F::Output> {
        f.as_mut().poll(&mut Context::from_waker(Waker::noop()))
    }

    #[test]
    fn issue_order_admission() {
        let b = ByteBudget::new(10, 10, AdmissionOrder::Issue);
        assert_eq!(
            poll(&mut Box::pin(b.acquire_strict(10, 5))),
            Poll::Ready(false)
        );

        let mut a3 = Box::pin(b.acquire_strict(4, 3));
        let mut a2 = Box::pin(b.acquire_strict(4, 2));
        let mut a1 = Box::pin(b.acquire_strict(4, 1));
        assert!(poll(&mut a3).is_pending() && poll(&mut a2).is_pending());
        assert!(poll(&mut a1).is_pending());
        // A 0-byte acquire without an issue number does not wait.
        assert_eq!(
            poll(&mut Box::pin(b.acquire_strict(0, 0))),
            Poll::Ready(false)
        );
        // A cancelled front leaves the queue without taking bytes.
        drop(a1);
        assert_eq!(b.queued.load(SeqCst), 2);

        b.release(2);
        // Fits, but a lower issue number waits: parks, counted.
        let mut a9 = Box::pin(b.acquire_strict(2, 9));
        assert!(poll(&mut a9).is_pending());
        assert_eq!(b.acquires_parked.load(Relaxed), 3);
        // No issue number: passes the queue when it fits.
        assert_eq!(
            poll(&mut Box::pin(b.acquire_strict(2, 0))),
            Poll::Ready(false)
        );

        b.release(6);
        assert_eq!(poll(&mut a2), Poll::Ready(true));
        assert!(poll(&mut a3).is_pending() && poll(&mut a9).is_pending());

        b.release(4);
        assert_eq!(poll(&mut a3), Poll::Ready(true));
        assert_eq!(poll(&mut a9), Poll::Ready(true));
        assert_eq!(b.queued.load(SeqCst), 0);
        assert_eq!(b.inflight_in_use(), 10);
    }
}
