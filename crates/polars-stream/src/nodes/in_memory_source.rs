use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, Morsel, MorselSeq};

pub struct InMemorySource {
    source: Arc<DataFrame>,
    morsel_size: usize,
    seq: AtomicU64,
}

impl InMemorySource {
    pub fn new(source: Arc<DataFrame>) -> Self {
        InMemorySource {
            source,
            morsel_size: 0,
            seq: AtomicU64::new(0),
        }
    }
}

impl ComputeNode for InMemorySource {
    fn initialize(&mut self, num_pipelines: usize) {
        let len = self.source.height();
        let ideal_block_count = (len / get_ideal_morsel_size()).max(1);
        let block_count = ideal_block_count.next_multiple_of(num_pipelines);
        self.morsel_size = len.div_ceil(block_count).max(1);
        self.seq = AtomicU64::new(0);
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.is_empty());
        let [mut send] = <[_; 1]>::try_from(send).ok().unwrap();

        scope.spawn_task(false, async move {
            let wait_group = WaitGroup::default();
            loop {
                let seq = self.seq.fetch_add(1, Ordering::Relaxed);
                let offset = (seq as usize * self.morsel_size) as i64;
                let df = self.source.slice(offset, self.morsel_size);
                if df.is_empty() {
                    break;
                }

                let mut morsel = Morsel::new(df, MorselSeq::new(seq));
                morsel.set_consume_token(wait_group.token());
                if send.send(morsel).await.is_err() {
                    break;
                }
                wait_group.wait().await;
            }

            Ok(())
        })
    }
}
