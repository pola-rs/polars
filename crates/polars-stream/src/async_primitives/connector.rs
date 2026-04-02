#![allow(unsafe_op_in_unsafe_fn)]
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll, Waker};

use atomic_waker::AtomicWaker;
use pin_project_lite::pin_project;

pub type Sender<T> = SenderExt<T, ()>;
pub type Receiver<T> = ReceiverExt<T, ()>;

/// Single-producer, single-consumer capacity-one channel.
pub fn connector<T>() -> (Sender<T>, Receiver<T>) {
    let connector = Arc::new(Connector::new(()));
    (
        Sender {
            connector: connector.clone(),
        },
        Receiver { connector },
    )
}

/// Single-producer, single-consumer capacity-one channel, with a shared common
/// value.
pub fn connector_with<T, S>(shared: S) -> (SenderExt<T, S>, ReceiverExt<T, S>) {
    let connector = Arc::new(Connector::new(shared));
    (
        SenderExt {
            connector: connector.clone(),
        },
        ReceiverExt { connector },
    )
}

/*
    For UnsafeCell safety, a sender may only set the FULL_BIT (giving exclusive
    access to value to the receiver), and a receiver may only unset the FULL_BIT
    (giving exclusive access back to the sender). Setting/clearing the FULL_BIT
    must be done with a Release ordering, and before reading/writing the value
    the FULL_BIT must be checked with an Acquire ordering.

    The exception is when the closed bit is set, at that point the unclosed
    end has full exclusive access.
*/

const FULL_BIT: u8 = 0b1;
const CLOSED_BIT: u8 = 0b10;
const WAITING_BIT: u8 = 0b100;

#[repr(align(128))]
struct Connector<T, S> {
    send_waker: AtomicWaker,
    recv_waker: AtomicWaker,
    value: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
    shared: S,
}

impl<T, S> Connector<T, S> {
    fn new(shared: S) -> Self {
        Self {
            send_waker: AtomicWaker::new(),
            recv_waker: AtomicWaker::new(),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(0),
            shared,
        }
    }
}

pub enum SendError<T> {
    Full(T),
    Closed(T),
}

pub enum RecvError {
    Empty,
    Closed,
}

// SAFETY: all the send methods may only be called from a single sender at a
// time, and similarly for all the recv methods from a single receiver.
impl<T, S> Connector<T, S> {
    unsafe fn poll_send(&self, value: &mut Option<T>, waker: &Waker) -> Poll<Result<(), T>> {
        if let Some(v) = value.take() {
            let mut state = self.state.load(Ordering::Acquire);
            if state & FULL_BIT == FULL_BIT {
                self.send_waker.register(waker);
                let (Ok(s) | Err(s)) = self.state.compare_exchange(
                    state,
                    state | WAITING_BIT,
                    Ordering::Relaxed,
                    Ordering::Acquire, // Receiver updated, re-acquire.
                );
                state = s;
            }

            match self.try_send_impl(v, state) {
                Ok(()) => {},
                Err(SendError::Closed(v)) => return Poll::Ready(Err(v)),
                Err(SendError::Full(v)) => {
                    *value = Some(v);
                    return Poll::Pending;
                },
            }
        }

        Poll::Ready(Ok(()))
    }

    unsafe fn try_send_impl(&self, value: T, state: u8) -> Result<(), SendError<T>> {
        if state & CLOSED_BIT == CLOSED_BIT {
            return Err(SendError::Closed(value));
        }
        if state & FULL_BIT == FULL_BIT {
            return Err(SendError::Full(value));
        }

        unsafe {
            self.value.get().write(MaybeUninit::new(value));
            let state = self.state.swap(FULL_BIT, Ordering::Release);
            if state & WAITING_BIT == WAITING_BIT {
                self.recv_waker.wake();
            }
            if state & CLOSED_BIT == CLOSED_BIT {
                // SAFETY: no synchronization needed, we are the only one left.
                // Restore the closed bit we just overwrote.
                self.state.store(CLOSED_BIT, Ordering::Relaxed);
                return Err(SendError::Closed(self.value.get().read().assume_init()));
            }
        }

        Ok(())
    }

    unsafe fn poll_recv(&self, waker: &Waker) -> Poll<Result<T, ()>> {
        let mut state = self.state.load(Ordering::Acquire);
        if state & FULL_BIT == 0 {
            self.recv_waker.register(waker);
            let (Ok(s) | Err(s)) = self.state.compare_exchange(
                state,
                state | WAITING_BIT,
                Ordering::Relaxed,
                Ordering::Acquire, // Sender updated, re-acquire.
            );
            state = s;
        }

        match self.try_recv_impl(state) {
            Ok(v) => Poll::Ready(Ok(v)),
            Err(RecvError::Empty) => Poll::Pending,
            Err(RecvError::Closed) => Poll::Ready(Err(())),
        }
    }

    unsafe fn try_recv_impl(&self, state: u8) -> Result<T, RecvError> {
        if state & FULL_BIT == FULL_BIT {
            unsafe {
                let ret = self.value.get().read().assume_init();
                let state = self.state.swap(0, Ordering::Release);
                if state & WAITING_BIT == WAITING_BIT {
                    self.send_waker.wake();
                }
                if state & CLOSED_BIT == CLOSED_BIT {
                    // Restore the closed bit we just overwrote.
                    self.state.store(CLOSED_BIT, Ordering::Relaxed);
                }
                return Ok(ret);
            }
        }

        // Check closed bit last so we do receive any last element sent before
        // closing sender.
        if state & CLOSED_BIT == CLOSED_BIT {
            return Err(RecvError::Closed);
        }

        Err(RecvError::Empty)
    }

    unsafe fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        self.try_send_impl(value, self.state.load(Ordering::Acquire))
    }

    unsafe fn try_recv(&self) -> Result<T, RecvError> {
        self.try_recv_impl(self.state.load(Ordering::Acquire))
    }

    /// # Safety
    /// You may not access this connector anymore as a sender after this call.
    unsafe fn close_send(&self) {
        self.state.fetch_or(CLOSED_BIT, Ordering::Relaxed);
        self.recv_waker.wake();
    }

    /// # Safety
    /// You may not access this connector anymore as a receiver after this call.
    unsafe fn close_recv(&self) {
        let state = self.state.fetch_or(CLOSED_BIT, Ordering::Acquire);
        drop(self.try_recv_impl(state));
        self.send_waker.wake();
    }
}

pub struct SenderExt<T, S> {
    connector: Arc<Connector<T, S>>,
}

unsafe impl<T: Send, S: Sync> Send for SenderExt<T, S> {}

impl<T, S> Drop for SenderExt<T, S> {
    fn drop(&mut self) {
        unsafe { self.connector.close_send() }
    }
}

pub struct ReceiverExt<T, S> {
    connector: Arc<Connector<T, S>>,
}

unsafe impl<T: Send, S: Sync> Send for ReceiverExt<T, S> {}

impl<T, S> Drop for ReceiverExt<T, S> {
    fn drop(&mut self) {
        unsafe { self.connector.close_recv() }
    }
}

pin_project! {
    pub struct SendFuture<'a, T, S> {
        connector: &'a Connector<T, S>,
        value: Option<T>,
    }
}

unsafe impl<T: Send, S: Sync> Send for SendFuture<'_, T, S> {}

impl<T: Send, S: Sync> SenderExt<T, S> {
    /// Returns a future that when awaited will send the value to the [`ReceiverExt`].
    /// Returns Err(value) if the connector is closed.
    #[must_use]
    pub fn send(&mut self, value: T) -> SendFuture<'_, T, S> {
        SendFuture {
            connector: &self.connector,
            value: Some(value),
        }
    }

    #[allow(unused)]
    pub fn try_send(&mut self, value: T) -> Result<(), SendError<T>> {
        unsafe { self.connector.try_send(value) }
    }

    pub fn shared(&self) -> &S {
        &self.connector.shared
    }
}

impl<T, S> std::future::Future for SendFuture<'_, T, S> {
    type Output = Result<(), T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(
            self.value.is_some(),
            "re-poll after Poll::Ready in connector SendFuture"
        );
        unsafe { self.connector.poll_send(self.project().value, cx.waker()) }
    }
}

pin_project! {
    pub struct RecvFuture<'a, T, S> {
        connector: &'a Connector<T, S>,
        done: bool,
    }
}

unsafe impl<T: Send, S: Sync> Send for RecvFuture<'_, T, S> {}

impl<T: Send, S: Sync> ReceiverExt<T, S> {
    /// Returns a future that when awaited will return `Ok(value)` once the
    /// value is received, or returns `Err(())` if the [`SenderExt`] was dropped
    /// before sending a value.
    #[must_use]
    pub fn recv(&mut self) -> RecvFuture<'_, T, S> {
        RecvFuture {
            connector: &self.connector,
            done: false,
        }
    }

    #[allow(unused)]
    pub fn try_recv(&mut self) -> Result<T, RecvError> {
        unsafe { self.connector.try_recv() }
    }

    pub fn shared(&self) -> &S {
        &self.connector.shared
    }
}

impl<T, S> std::future::Future for RecvFuture<'_, T, S> {
    type Output = Result<T, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(
            !self.done,
            "re-poll after Poll::Ready in connector SendFuture"
        );
        unsafe { self.connector.poll_recv(cx.waker()) }
    }
}
