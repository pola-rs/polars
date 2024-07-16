use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use atomic_waker::AtomicWaker;
use pin_project_lite::pin_project;

/// Single-producer, single-consumer capacity-one channel.
pub fn connector<T>() -> (Sender<T>, Receiver<T>) {
    let connector = Arc::new(Connector::default());
    (
        Sender {
            connector: connector.clone(),
        },
        Receiver { connector },
    )
}

/*
    For UnsafeCell safety, a sender may only set the FULL_BIT (giving exclusive
    access to value to the receiver), and a receiver may only unset the FULL_BIT
    (giving exclusive access back to the sender).

    The exception is when the closed bit is set, at that point the unclosed
    end has full exclusive access.
*/

const FULL_BIT: u8 = 0b1;
const CLOSED_BIT: u8 = 0b10;
const WAITING_BIT: u8 = 0b100;

#[repr(align(64))]
struct Connector<T> {
    send_waker: AtomicWaker,
    recv_waker: AtomicWaker,
    value: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
}

impl<T> Default for Connector<T> {
    fn default() -> Self {
        Self {
            send_waker: AtomicWaker::new(),
            recv_waker: AtomicWaker::new(),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(0),
        }
    }
}

impl<T> Drop for Connector<T> {
    fn drop(&mut self) {
        if self.state.load(Ordering::Acquire) & FULL_BIT == FULL_BIT {
            unsafe {
                self.value.get().drop_in_place();
            }
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
impl<T> Connector<T> {
    unsafe fn poll_send(&self, value: &mut Option<T>, waker: &Waker) -> Poll<Result<(), T>> {
        if let Some(v) = value.take() {
            let mut state = self.state.load(Ordering::Relaxed);
            if state & FULL_BIT == FULL_BIT {
                self.send_waker.register(waker);
                let (Ok(s) | Err(s)) = self.state.compare_exchange(
                    state,
                    state | WAITING_BIT,
                    Ordering::Release,
                    Ordering::Relaxed,
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
            let state = self.state.swap(FULL_BIT, Ordering::AcqRel);
            if state & WAITING_BIT == WAITING_BIT {
                self.recv_waker.wake();
            }
            if state & CLOSED_BIT == CLOSED_BIT {
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
                Ordering::Release,
                Ordering::Acquire,
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
                let state = self.state.swap(0, Ordering::Acquire);
                if state & WAITING_BIT == WAITING_BIT {
                    self.send_waker.wake();
                }
                if state & CLOSED_BIT == CLOSED_BIT {
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
        self.try_send_impl(value, self.state.load(Ordering::Relaxed))
    }

    unsafe fn try_recv(&self) -> Result<T, RecvError> {
        self.try_recv_impl(self.state.load(Ordering::Acquire))
    }

    /// # Safety
    /// After calling close as a sender/receiver, you may not access
    /// this connector anymore as that end.
    unsafe fn close(&self) {
        self.state.fetch_or(CLOSED_BIT, Ordering::Relaxed);
        self.send_waker.wake();
        self.recv_waker.wake();
    }
}

pub struct Sender<T> {
    connector: Arc<Connector<T>>,
}

unsafe impl<T: Send> Send for Sender<T> {}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        unsafe { self.connector.close() }
    }
}

pub struct Receiver<T> {
    connector: Arc<Connector<T>>,
}

unsafe impl<T: Send> Send for Receiver<T> {}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        unsafe { self.connector.close() }
    }
}

pin_project! {
    pub struct SendFuture<'a, T> {
        connector: &'a Connector<T>,
        value: Option<T>,
    }
}

unsafe impl<'a, T: Send> Send for SendFuture<'a, T> {}

impl<T: Send> Sender<T> {
    /// Returns a future that when awaited will send the value to the [`Receiver`].
    /// Returns Err(value) if the connector is closed.
    #[must_use]
    pub fn send(&mut self, value: T) -> SendFuture<'_, T> {
        SendFuture {
            connector: &self.connector,
            value: Some(value),
        }
    }

    #[allow(unused)]
    pub fn try_send(&mut self, value: T) -> Result<(), SendError<T>> {
        unsafe { self.connector.try_send(value) }
    }
}

impl<T> std::future::Future for SendFuture<'_, T> {
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
    pub struct RecvFuture<'a, T> {
        connector: &'a Connector<T>,
        done: bool,
    }
}

unsafe impl<'a, T: Send> Send for RecvFuture<'a, T> {}

impl<T: Send> Receiver<T> {
    /// Returns a future that when awaited will return `Ok(value)` once the
    /// value is received, or returns `Err(())` if the [`Sender`] was dropped
    /// before sending a value.
    #[must_use]
    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture {
            connector: &self.connector,
            done: false,
        }
    }

    #[allow(unused)]
    pub fn try_recv(&mut self) -> Result<T, RecvError> {
        unsafe { self.connector.try_recv() }
    }
}

impl<T> std::future::Future for RecvFuture<'_, T> {
    type Output = Result<T, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(
            !self.done,
            "re-poll after Poll::Ready in connector SendFuture"
        );
        unsafe { self.connector.poll_recv(cx.waker()) }
    }
}
