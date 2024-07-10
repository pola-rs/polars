use std::cmp::Reverse;
use std::collections::BinaryHeap;

use polars_utils::priority::Priority;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::morsel::{Morsel, MorselSeq};

/// Stores the state for which inserter we need to poll.
enum PollState {
    NoPoll,
    Poll(usize),
    PollAll,
}

pub struct Linearizer {
    receivers: Vec<Receiver<Morsel>>,
    poll_state: PollState,

    heap: BinaryHeap<Priority<Reverse<MorselSeq>, (usize, Morsel)>>,
}

impl Linearizer {
    pub fn new(num_inserters: usize, buffer_size: usize) -> (Self, Vec<Inserter>) {
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
            heap: BinaryHeap::default(),
        };
        (slf, inserters)
    }

    pub async fn get(&mut self) -> Option<Morsel> {
        // The idea is that we have exactly one morsel per inserter in the
        // binary heap, and when we take one out we must refill it. This way we
        // always ensure we have the morsel with the lowest global sequence id.
        let poll_range = match self.poll_state {
            PollState::NoPoll => 0..0,
            PollState::Poll(i) => i..i + 1,
            PollState::PollAll => 0..self.receivers.len(),
        };
        for recv_idx in poll_range {
            // If no morsel was received from that particular inserter, that
            // stream is done and thus we no longer need to consider it for the
            // global order.
            if let Some(morsel) = self.receivers[recv_idx].recv().await {
                self.heap
                    .push(Priority(Reverse(morsel.seq()), (recv_idx, morsel)));
            }
        }

        if let Some(first_in_merged_streams) = self.heap.pop() {
            let (receiver_idx, morsel) = first_in_merged_streams.1;
            self.poll_state = PollState::Poll(receiver_idx);
            Some(morsel)
        } else {
            self.poll_state = PollState::NoPoll;
            None
        }
    }
}

pub struct Inserter {
    sender: Sender<Morsel>,
}

impl Inserter {
    pub async fn insert(&mut self, mut morsel: Morsel) -> Result<(), Morsel> {
        // Drop the consume token, but only after the send has succeeded. This
        // ensures we have backpressure, but only once the channel fills up.
        let consume_token = morsel.take_consume_token();
        self.sender.send(morsel).await.map_err(|e| e.0)?;
        drop(consume_token);
        Ok(())
    }
}
