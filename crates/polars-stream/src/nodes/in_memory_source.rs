use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::compute_node_prelude::*;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, MorselSeq, SourceToken};

pub struct InMemorySourceNode {
    source: Option<Arc<DataFrame>>,
    morsel_size: usize,
    seq: AtomicU64,
    seq_offset: MorselSeq,
}

impl InMemorySourceNode {
    pub fn new(source: Arc<DataFrame>, seq_offset: MorselSeq) -> Self {
        InMemorySourceNode {
            source: Some(source),
            morsel_size: 0,
            seq: AtomicU64::new(0),
            seq_offset,
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

        // As a temporary hack for some nodes (like the FunctionIR::FastCount)
        // node that rely on an empty input, always ensure we send at least one
        // morsel.
        // TODO: remove this hack.
        let exhausted = if let Some(src) = &self.source {
            let seq = self.seq.load(Ordering::Relaxed);
            seq > 0 && seq * self.morsel_size as u64 >= src.height() as u64
        } else {
            true
        };
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
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);
        let senders = send_ports[0].take().unwrap().parallel();
        let source = self.source.as_ref().unwrap();

        // TODO: can this just be serial, using the work distributor?
        let source_token = SourceToken::new();
        for mut send in senders {
            let slf = &*self;
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let wait_group = WaitGroup::default();
                loop {
                    let seq = slf.seq.fetch_add(1, Ordering::Relaxed);
                    let offset = (seq as usize * slf.morsel_size) as i64;
                    let df = source.slice(offset, slf.morsel_size);

                    // TODO: remove this 'always sent at least one morsel'
                    // condition, see update_state.
                    if df.is_empty() && seq > 0 {
                        break;
                    }

                    let morsel_seq = MorselSeq::new(seq).offset_by(slf.seq_offset);
                    let mut morsel = Morsel::new(df, morsel_seq, source_token.clone());
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
