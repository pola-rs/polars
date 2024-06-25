use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;

#[derive(Default)]
pub struct TaskParker {
    state: AtomicU8,
    waker: Mutex<Option<Waker>>,
}

impl TaskParker {
    const RUNNING: u8 = 0;
    const PREPARING_TO_PARK: u8 = 1;
    const PARKED: u8 = 2;

    /// Returns a future that when awaited parks this task.
    ///
    /// Any notifications between calls to park and the await will cancel
    /// the park attempt.
    pub fn park(&self) -> TaskParkFuture<'_> {
        self.state.store(Self::PREPARING_TO_PARK, Ordering::SeqCst);
        TaskParkFuture { parker: self }
    }

    /// Unparks the parked task, if it was parked.
    pub fn unpark(&self) {
        let state = self.state.load(Ordering::SeqCst);
        if state != Self::RUNNING {
            let old_state = self.state.swap(Self::RUNNING, Ordering::SeqCst);
            if old_state == Self::PARKED {
                if let Some(w) = self.waker.lock().take() {
                    w.wake();
                }
            }
        }
    }
}

pub struct TaskParkFuture<'a> {
    parker: &'a TaskParker,
}

impl<'a> Future for TaskParkFuture<'a> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.parker.state.load(Ordering::SeqCst);
        loop {
            match state {
                TaskParker::RUNNING => return Poll::Ready(()),

                TaskParker::PARKED => {
                    // Refresh our waker.
                    match &mut *self.parker.waker.lock() {
                        Some(w) => w.clone_from(cx.waker()),
                        None => return Poll::Ready(()), // Apparently someone woke us up.
                    }
                },
                TaskParker::PREPARING_TO_PARK => {
                    // Install waker first before publishing that we're parked
                    // to prevent missed notifications.
                    *self.parker.waker.lock() = Some(cx.waker().clone());
                    match self.parker.state.compare_exchange_weak(
                        TaskParker::PREPARING_TO_PARK,
                        TaskParker::PARKED,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => return Poll::Pending,
                        Err(s) => state = s,
                    }
                },
                _ => unreachable!(),
            }
        }
    }
}
