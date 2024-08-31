use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::compute_node_prelude::*;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, MorselSeq, SourceToken};

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
        let ideal_morsel_count = (len / get_ideal_morsel_size()).max(1);
        let morsel_count = ideal_morsel_count.next_multiple_of(num_pipelines);
        self.morsel_size = len.div_ceil(morsel_count).max(1);
        self.seq = AtomicU64::new(0);
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
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
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.is_empty() && send.len() == 1);
        let senders = send[0].take().unwrap().parallel();
        let source = self.source.as_ref().unwrap();

        // TODO: can this just be serial, using the work distributor?
        for mut send in senders {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let wait_group = WaitGroup::default();
                let source_token = SourceToken::new();
                loop {
                    let seq = slf.seq.fetch_add(1, Ordering::Relaxed);
                    let offset = (seq as usize * slf.morsel_size) as i64;
                    let df = source.slice(offset, slf.morsel_size);
                    if df.is_empty() {
                        break;
                    }

                    let mut morsel = Morsel::new(df, MorselSeq::new(seq), source_token.clone());
                    morsel.set_consume_token(wait_group.token());
                    if send.send(morsel).await.is_err() {
                        break;
                    }

                    wait_group.wait().await;
                    if source_token.stop_requested() {
                        break;
                    }
                }

                Ok(())
            }));
        }
    }
}
