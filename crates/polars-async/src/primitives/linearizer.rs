use std::collections::BinaryHeap;

use tokio::sync::mpsc::{Receiver, Sender, channel};

/// Stores the state for which inserter we need to poll.
enum PollState {
    NoPoll,
    Poll(usize),
    PollAll,
}

struct LinearedItem<T> {
    value: T,
    sender_id: usize,
}

impl<T: Ord> PartialEq for LinearedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}
impl<T: Ord> Eq for LinearedItem<T> {}
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<T: Ord> PartialOrd for LinearedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}
impl<T: Ord> Ord for LinearedItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

/// Utility to convert the input of `N` senders of ordered data into `1` stream of ordered data.
pub struct Linearizer<T> {
    receivers: Vec<Receiver<T>>,
    poll_state: PollState,

    heap: BinaryHeap<LinearedItem<T>>,
}

impl<T: Ord> Linearizer<T> {
    pub fn new(num_inserters: usize, buffer_size: usize) -> (Self, Vec<Inserter<T>>) {
        let mut receivers = Vec::with_capacity(num_inserters);
        let mut inserters = Vec::with_capacity(num_inserters);

        for _ in 0..num_inserters {
            // We could perhaps use a bespoke spsc bounded channel here in the
            // future, instead of tokio's mpsc channel.
            let (sender, receiver) = channel(buffer_size);
            receivers.push(receiver);
            inserters.push(Inserter { sender });
        }
        let slf = Self {
            receivers,
            poll_state: PollState::PollAll,
            heap: BinaryHeap::with_capacity(num_inserters),
        };
        (slf, inserters)
    }

    pub fn new_with_maintain_order(
        num_inserters: usize,
        buffer_size: usize,
        maintain_order: bool,
    ) -> (Self, Vec<Inserter<T>>) {
        if maintain_order {
            return Self::new(num_inserters, buffer_size);
        }

        let (sender, receiver) = channel(buffer_size * num_inserters);
        let receivers = vec![receiver];
        let inserters = (0..num_inserters)
            .map(|_| Inserter {
                sender: sender.clone(),
            })
            .collect();

        let slf = Self {
            receivers,
            poll_state: PollState::PollAll,
            heap: BinaryHeap::with_capacity(1),
        };
        (slf, inserters)
    }

    /// Fetch the next ordered item produced by senders.
    ///
    /// This may wait for at each sender to have sent at least one value before the [`Linearizer`]
    /// starts producing.
    ///
    /// If all senders have closed their channels and there are no more buffered values, this
    /// returns `None`.
    pub async fn get(&mut self) -> Option<T> {
        // The idea is that we have exactly one value per inserter in the
        // binary heap, and when we take one out we must refill it. This way we
        // always ensure we have the value with the highest global order.
        let poll_range = match self.poll_state {
            PollState::NoPoll => 0..0,
            PollState::Poll(i) => i..i + 1,
            PollState::PollAll => 0..self.receivers.len(),
        };

        for sender_id in poll_range {
            // If no value was received from that particular inserter, that
            // stream is done and thus we no longer need to consider it for the
            // global order.
            if let Some(value) = self.receivers[sender_id].recv().await {
                self.heap.push(LinearedItem { value, sender_id });
            }
        }

        if let Some(first_in_merged_streams) = self.heap.pop() {
            let LinearedItem { value, sender_id } = first_in_merged_streams;
            self.poll_state = PollState::Poll(sender_id);
            Some(value)
        } else {
            self.poll_state = PollState::NoPoll;
            None
        }
    }
}

pub struct Inserter<T> {
    sender: Sender<T>,
}

impl<T: Ord> Inserter<T> {
    pub async fn insert(&mut self, value: T) -> Result<(), T> {
        self.sender.send(value).await.map_err(|e| e.0)
    }
}
