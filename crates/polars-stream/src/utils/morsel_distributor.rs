use std::sync::Arc;

use polars_async::primitives::distributor_channel::{Receiver, Sender, distributor_channel};
use polars_utils::relaxed_cell::RelaxedCell;

use crate::morsel::{Morsel, MorselSeq};

/// A [`distributor_channel`] for morsels (and an associated payload per morsel)
/// which relabels the morsel sequence ids to be unique before distributing.
///
/// Within a single pipe consecutive sequence ids may repeat, but when
/// distributing morsels over several pipes this would destroy the order: the
/// relative order of two morsels with the same sequence id is lost as soon as
/// they end up in different pipes.
pub fn morsel_distributor<T>(
    num_receivers: usize,
    bufsize: usize,
    seq_offset: Arc<RelaxedCell<u64>>,
) -> (MorselDistributorSender<T>, Vec<Receiver<(Morsel, T)>>) {
    let (sender, receivers) = distributor_channel(num_receivers, bufsize);
    let sender = MorselDistributorSender {
        sender,
        offset: seq_offset.load(),
        seq_offset,
        prev_orig_seq: None,
    };
    (sender, receivers)
}

pub struct MorselDistributorSender<T> {
    sender: Sender<(Morsel, T)>,
    seq_offset: Arc<RelaxedCell<u64>>,
    offset: u64,
    prev_orig_seq: Option<MorselSeq>,
}

impl<T: Send> MorselDistributorSender<T> {
    pub async fn send(&mut self, (mut morsel, payload): (Morsel, T)) -> Result<(), (Morsel, T)> {
        let orig_seq = morsel.seq();
        if Some(orig_seq) == self.prev_orig_seq {
            self.offset += 1;
            self.seq_offset.store(self.offset);
        }
        self.prev_orig_seq = Some(orig_seq);
        morsel.set_seq(orig_seq.offset_by_u64(self.offset));

        // Important: we have to drop the consume token before
        // going into the buffered distributor.
        drop(morsel.take_consume_token());
        self.sender.send((morsel, payload)).await
    }
}
