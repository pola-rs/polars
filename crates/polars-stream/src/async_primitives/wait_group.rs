use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

#[derive(Default, Debug)]
struct WaitGroupInner {
    waker: Mutex<Option<Waker>>,
    token_count: AtomicUsize,
    is_waiting: AtomicBool,
}

#[derive(Default)]
pub struct WaitGroup {
    inner: Arc<WaitGroupInner>,
}

impl WaitGroup {
    /// Creates a token.
    pub fn token(&self) -> WaitToken {
        self.inner.token_count.fetch_add(1, Ordering::Relaxed);
        WaitToken {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Waits until all created tokens are dropped.
    ///
    /// # Panics
    /// Panics if there is more than one simultaneous waiter.
    pub async fn wait(&self) {
        let was_waiting = self.inner.is_waiting.swap(true, Ordering::Relaxed);
        assert!(!was_waiting);
        WaitGroupFuture { inner: &self.inner }.await
    }
}

struct WaitGroupFuture<'a> {
    inner: &'a Arc<WaitGroupInner>,
}

impl Future for WaitGroupFuture<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.inner.token_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        // Check count again while holding lock to prevent missed notifications.
        let mut waker_lock = self.inner.waker.lock();
        if self.inner.token_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        let waker = cx.waker().clone();
        *waker_lock = Some(waker);
        Poll::Pending
    }
}

impl<'a> Drop for WaitGroupFuture<'a> {
    fn drop(&mut self) {
        self.inner.is_waiting.store(false, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct WaitToken {
    inner: Arc<WaitGroupInner>,
}

impl Clone for WaitToken {
    fn clone(&self) -> Self {
        self.inner.token_count.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Drop for WaitToken {
    fn drop(&mut self) {
        // Token count was 1, we must notify.
        if self.inner.token_count.fetch_sub(1, Ordering::Release) == 1 {
            if let Some(w) = self.inner.waker.lock().take() {
                w.wake();
            }
        }
    }
}
