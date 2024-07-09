use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::compute_node_prelude::*;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, MorselSeq};

pub struct InMemorySourceNode {
    source: Option<Arc<DataFrame>>,
    morsel_size: usize,
    seq: AtomicU64,
}

impl InMemorySourceNode {
    pub fn new(source: Arc<DataFrame>) -> Self {
        InMemorySourceNode {
            source: Some(source),
            morsel_size: 0,
            seq: AtomicU64::new(0),
        }
    }
}

impl ComputeNode for InMemorySourceNode {
    fn name(&self) -> &str {
        "in_memory_source"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        let len = self.source.as_ref().unwrap().height();
        let ideal_block_count = (len / get_ideal_morsel_size()).max(1);
        let block_count = ideal_block_count.next_multiple_of(num_pipelines);
        self.morsel_size = len.div_ceil(block_count).max(1);
        self.seq = AtomicU64::new(0);
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(recv.is_empty());
        assert!(send.len() == 1);

        let exhausted = self
            .source
            .as_ref()
            .map(|s| {
                self.seq.load(Ordering::Relaxed) * self.morsel_size as u64 >= s.height() as u64
            })
            .unwrap_or(true);

        if send[0] == PortState::Done || exhausted {
            send[0] = PortState::Done;
            self.source = None;
        } else {
            send[0] = PortState::Ready;
        }
    }

    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        _state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>> {
        assert!(recv.is_empty() && send.len() == 1);
        let mut send = send[0].take().unwrap();
        let source = self.source.as_ref().unwrap();

        scope.spawn_task(TaskPriority::Low, async move {
            let wait_group = WaitGroup::default();
            loop {
                let seq = self.seq.fetch_add(1, Ordering::Relaxed);
                let offset = (seq as usize * self.morsel_size) as i64;
                let df = source.slice(offset, self.morsel_size);
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
