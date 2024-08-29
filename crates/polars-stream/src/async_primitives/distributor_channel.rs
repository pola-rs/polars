use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_utils::CachePadded;
use rand::prelude::*;

use super::task_parker::TaskParker;

/// Single-producer multi-consumer FIFO channel.
///
/// Each [`Receiver`] has an internal buffer of `bufsize`. Thus it is possible
/// that when one [`Sender`] is exhausted some other receivers still have data
/// available.
///
/// The FIFO order is only guaranteed per receiver. That is, each receiver is
/// guaranteed to see a subset of the data sent by the sender in the order the
/// sender sent it in, but not necessarily contiguously.
pub fn distributor_channel<T>(
    num_receivers: usize,
    bufsize: usize,
) -> (Sender<T>, Vec<Receiver<T>>) {
    let capacity = bufsize.next_power_of_two();
    let receivers = (0..num_receivers)
        .map(|_| {
            CachePadded::new(ReceiverSlot {
                closed: AtomicBool::new(false),
                read_head: AtomicUsize::new(0),
                parker: TaskParker::default(),
                data: (0..capacity)
                    .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                    .collect(),
            })
        })
        .collect();
    let inner = Arc::new(DistributorInner {
        send_closed: AtomicBool::new(false),
        send_parker: TaskParker::default(),
        write_heads: (0..num_receivers).map(|_| AtomicUsize::new(0)).collect(),
        receivers,

        bufsize,
        mask: capacity - 1,
    });

    let receivers = (0..num_receivers)
        .map(|index| Receiver {
            inner: inner.clone(),
            index,
        })
        .collect();

    let sender = Sender {
        inner,
        round_robin_idx: 0,
        rng: SmallRng::from_rng(&mut rand::thread_rng()).unwrap(),
    };

    (sender, receivers)
}

pub enum SendError<T> {
    Full(T),
    Closed(T),
}

pub enum RecvError {
    Empty,
    Closed,
}

struct ReceiverSlot<T> {
    closed: AtomicBool,
    read_head: AtomicUsize,
    parker: TaskParker,
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

struct DistributorInner<T> {
    send_closed: AtomicBool,
    send_parker: TaskParker,
    write_heads: Vec<AtomicUsize>,
    receivers: Vec<CachePadded<ReceiverSlot<T>>>,

    bufsize: usize,
    mask: usize,
}

impl<T> DistributorInner<T> {
    fn reduce_index(&self, idx: usize) -> usize {
        idx & self.mask
    }
}

pub struct Sender<T> {
    inner: Arc<DistributorInner<T>>,
    round_robin_idx: usize,
    rng: SmallRng,
}

pub struct Receiver<T> {
    inner: Arc<DistributorInner<T>>,
    index: usize,
}

unsafe impl<T: Send> Send for Sender<T> {}
unsafe impl<T: Send> Send for Receiver<T> {}

impl<T: Send> Sender<T> {
    pub async fn send(&mut self, mut value: T) -> Result<(), T> {
        let num_receivers = self.inner.receivers.len();
        loop {
            // Fast-path.
            self.round_robin_idx += 1;
            if self.round_robin_idx >= num_receivers {
                self.round_robin_idx -= num_receivers;
            }

            let mut hungriest_idx = self.round_robin_idx;
            let mut shortest_len = self.upper_bound_len(self.round_robin_idx);
            for _ in 0..4 {
                let idx = ((self.rng.gen::<u32>() as u64 * num_receivers as u64) >> 32) as usize;
                let len = self.upper_bound_len(idx);
                if len < shortest_len {
                    shortest_len = len;
                    hungriest_idx = idx;
                }
            }

            match self.try_send(hungriest_idx, value) {
                Ok(()) => return Ok(()),
                Err(SendError::Full(v)) => value = v,
                Err(SendError::Closed(v)) => value = v,
            }

            // Do one proper search before parking.
            let park = self.inner.send_parker.park();

            // Try all receivers, starting at a random index.
            let mut idx = ((self.rng.gen::<u32>() as u64 * num_receivers as u64) >> 32) as usize;
            let mut all_closed = true;
            for _ in 0..num_receivers {
                match self.try_send(idx, value) {
                    Ok(()) => return Ok(()),
                    Err(SendError::Full(v)) => {
                        all_closed = false;
                        value = v;
                    },
                    Err(SendError::Closed(v)) => value = v,
                }

                idx += 1;
                if idx >= num_receivers {
                    idx -= num_receivers;
                }
            }

            if all_closed {
                return Err(value);
            }

            park.await;
        }
    }

    fn upper_bound_len(&self, recv_idx: usize) -> usize {
        let read_head = self.inner.receivers[recv_idx]
            .read_head
            .load(Ordering::SeqCst);
        let write_head = self.inner.write_heads[recv_idx].load(Ordering::Relaxed);
        write_head.wrapping_sub(read_head)
    }

    fn try_send(&self, recv_idx: usize, value: T) -> Result<(), SendError<T>> {
        let read_head = self.inner.receivers[recv_idx]
            .read_head
            .load(Ordering::SeqCst);
        let write_head = self.inner.write_heads[recv_idx].load(Ordering::Relaxed);
        let len = write_head.wrapping_sub(read_head);
        if len < self.inner.bufsize {
            let idx = self.inner.reduce_index(write_head);
            unsafe {
                self.inner.receivers[recv_idx].data[idx]
                    .get()
                    .write(MaybeUninit::new(value));
                self.inner.write_heads[recv_idx]
                    .store(write_head.wrapping_add(1), Ordering::SeqCst);
            }
            self.inner.receivers[recv_idx].parker.unpark();
            Ok(())
        } else if self.inner.receivers[recv_idx].closed.load(Ordering::SeqCst) {
            Err(SendError::Closed(value))
        } else {
            Err(SendError::Full(value))
        }
    }
}

impl<T: Send> Receiver<T> {
    /// Note: This intentionally takes `&mut` to ensure it is only accessed in a single-threaded
    /// manner.
    pub async fn recv(&mut self) -> Result<T, ()> {
        loop {
            // Fast-path.
            match self.try_recv() {
                Ok(v) => return Ok(v),
                Err(RecvError::Closed) => return Err(()),
                Err(RecvError::Empty) => {},
            }

            // Try again, threatening to park if there's still nothing.
            let park = self.inner.receivers[self.index].parker.park();
            match self.try_recv() {
                Ok(v) => return Ok(v),
                Err(RecvError::Closed) => return Err(()),
                Err(RecvError::Empty) => {},
            }
            park.await;
        }
    }

    fn try_recv(&self) -> Result<T, RecvError> {
        let read_head = self.inner.receivers[self.index]
            .read_head
            .load(Ordering::Relaxed);
        let write_head = self.inner.write_heads[self.index].load(Ordering::SeqCst);
        if read_head != write_head {
            let idx = self.inner.reduce_index(read_head);
            let read;
            unsafe {
                let ptr = self.inner.receivers[self.index].data[idx].get();
                read = ptr.read().assume_init();
                self.inner.receivers[self.index]
                    .read_head
                    .store(read_head.wrapping_add(1), Ordering::SeqCst);
            }
            self.inner.send_parker.unpark();
            Ok(read)
        } else if self.inner.send_closed.load(Ordering::SeqCst) {
            Err(RecvError::Closed)
        } else {
            Err(RecvError::Empty)
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.inner.send_closed.store(true, Ordering::SeqCst);
        for recv in &self.inner.receivers {
            recv.parker.unpark();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.receivers[self.index]
            .closed
            .store(true, Ordering::SeqCst);
        self.inner.send_parker.unpark();
    }
}

impl<T> Drop for DistributorInner<T> {
    fn drop(&mut self) {
        for r in 0..self.receivers.len() {
            // We have exclusive access, so we only need to atomically load once.
            let write_head = self.write_heads[r].load(Ordering::SeqCst);
            let mut read_head = self.receivers[r].read_head.load(Ordering::Relaxed);
            while read_head != write_head {
                let idx = self.reduce_index(read_head);
                unsafe {
                    (*self.receivers[r].data[idx].get()).assume_init_drop();
                }
                read_head = read_head.wrapping_add(1);
            }
        }
    }
}
