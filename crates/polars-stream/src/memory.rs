use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

/// Tracks memory usage in the streaming engine and provides backpressure
/// when a configured limit is exceeded.
///
/// When no limit is set (`limit == 0`), all operations are effectively no-ops
/// with minimal overhead (a single branch on `has_limit()`).
pub struct MemoryTracker {
    used: AtomicU64,
    limit: u64,
    waiters: Mutex<Vec<Waker>>,
}

impl MemoryTracker {
    pub fn new(limit: u64) -> Self {
        Self {
            used: AtomicU64::new(0),
            limit,
            waiters: Mutex::new(Vec::new()),
        }
    }

    /// Returns true if a memory limit has been configured.
    #[inline]
    pub fn has_limit(&self) -> bool {
        self.limit > 0
    }

    /// Record `bytes` as allocated. Short-circuits if no limit is set.
    #[inline]
    pub fn alloc(&self, bytes: u64) {
        if !self.has_limit() {
            return;
        }
        self.used.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record `bytes` as freed. Wakes any tasks waiting for memory
    /// if usage drops below the limit. Short-circuits if no limit is set.
    #[inline]
    pub fn free(&self, bytes: u64) {
        if !self.has_limit() {
            return;
        }
        let prev = self.used.fetch_sub(bytes, Ordering::Relaxed);
        // If we crossed from over-limit to under-limit, wake all waiters.
        if prev > self.limit && prev - bytes <= self.limit {
            let waiters: Vec<Waker> = {
                let mut guard = self.waiters.lock();
                std::mem::take(&mut *guard)
            };
            for waker in waiters {
                waker.wake();
            }
        }
    }

    /// Async wait until memory usage is at or below the limit.
    /// Resolves immediately if no limit is set or if usage is under the limit.
    pub fn wait_for_available(&self) -> WaitForAvailable<'_> {
        WaitForAvailable { tracker: self }
    }

    /// Returns the current tracked memory usage in bytes.
    #[allow(unused)]
    pub fn current_usage(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
}

pub struct WaitForAvailable<'a> {
    tracker: &'a MemoryTracker,
}

impl Future for WaitForAvailable<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if !self.tracker.has_limit() {
            return Poll::Ready(());
        }
        if self.tracker.used.load(Ordering::Relaxed) <= self.tracker.limit {
            return Poll::Ready(());
        }
        // Over limit: register waker and park.
        self.tracker.waiters.lock().push(cx.waker().clone());
        // Re-check after registering to avoid lost wakeups.
        if self.tracker.used.load(Ordering::Relaxed) <= self.tracker.limit {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_no_limit() {
        let tracker = MemoryTracker::new(0);
        assert!(!tracker.has_limit());
        // alloc/free should be no-ops, no panics.
        tracker.alloc(1_000_000);
        assert_eq!(tracker.current_usage(), 0);
        tracker.free(500_000);
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_alloc_and_free() {
        let tracker = MemoryTracker::new(1000);
        assert!(tracker.has_limit());

        tracker.alloc(400);
        assert_eq!(tracker.current_usage(), 400);

        tracker.alloc(300);
        assert_eq!(tracker.current_usage(), 700);

        tracker.free(200);
        assert_eq!(tracker.current_usage(), 500);

        tracker.free(500);
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_wait_resolves_immediately_no_limit() {
        let tracker = MemoryTracker::new(0);
        tracker.alloc(999_999); // no-op since no limit
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            // Should resolve immediately.
            tracker.wait_for_available().await;
        });
    }

    #[test]
    fn test_wait_resolves_immediately_under_limit() {
        let tracker = MemoryTracker::new(1000);
        tracker.alloc(500);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            tracker.wait_for_available().await;
        });
    }

    #[test]
    fn test_wait_parks_then_wakes_on_free() {
        let tracker = Arc::new(MemoryTracker::new(1000));
        tracker.alloc(1500); // over limit

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let tracker2 = tracker.clone();
        rt.block_on(async {
            let handle = tokio::spawn({
                let t = tracker2.clone();
                async move {
                    // This should park until memory is freed.
                    t.wait_for_available().await;
                }
            });

            // Give the spawned task time to park.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            // Free enough memory to go under limit.
            tracker2.free(600); // 1500 - 600 = 900 <= 1000

            // The handle should now resolve.
            tokio::time::timeout(std::time::Duration::from_secs(2), handle)
                .await
                .expect("timed out waiting for task to wake")
                .expect("task panicked");
        });
    }
}
