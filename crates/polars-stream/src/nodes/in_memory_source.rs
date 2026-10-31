use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_async::primitives::wait_group::WaitGroup;

use super::compute_node_prelude::*;
use crate::morsel::{MorselSeq, SourceToken, get_ideal_morsel_size};

#[derive(Debug)]
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

    pub fn new_no_morsel_split(source: Arc<DataFrame>, seq_offset: MorselSeq) -> Self {
        let morsel_size = source.height();

        InMemorySourceNode {
            source: Some(source),
            morsel_size,
            seq: AtomicU64::new(0),
            seq_offset,
        }
    }
}

impl ComputeNode for InMemorySourceNode {
    fn name(&self) -> &str {
        "in-memory-source"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert!(send.len() == 1);

        if self.morsel_size == 0 {
            let len = self.source.as_ref().unwrap().height();
            self.morsel_size = in_memory_morsel_size(len, state.num_pipelines);
            self.seq = AtomicU64::new(0);
        }

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

    fn memory_usage(&self) -> NodeMemoryUsage {
        match &self.source {
            // Finishing only frees the frame if no one else holds a reference to it.
            Some(df) if Arc::strong_count(df) == 1 => NodeMemoryUsage::HoldingUntilDone,
            _ => NodeMemoryUsage::Bounded,
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);
        let senders = send_ports[0].take().unwrap().parallel();
        let source = self.source.as_ref().unwrap();
        let morsel_size = self.morsel_size;

        // TODO: can this just be serial, using the work distributor?
        spawn_in_memory_senders(
            scope,
            senders,
            &self.seq,
            self.seq_offset,
            move |seq| {
                let df = source.slice((seq as usize * morsel_size) as i64, morsel_size);
                // TODO: remove this 'always sent at least one morsel'
                // condition, see update_state.
                std::future::ready((df.height() > 0 || seq == 0).then_some(df))
            },
            join_handles,
        );
    }
}

/// The size of the morsels to split `len` rows in memory into.
pub fn in_memory_morsel_size(len: usize, num_pipelines: usize) -> usize {
    let ideal_morsel_count = (len / get_ideal_morsel_size()).max(1);
    let morsel_count = ideal_morsel_count.next_multiple_of(num_pipelines);
    len.div_ceil(morsel_count).max(1)
}

/// Spawns a task for every sender. Each task takes the next sequence number from `seq` and sends
/// the frame that `get_df` gives for it, until `get_df` gives `None`.
pub fn spawn_in_memory_senders<'env, 's, F, Fut>(
    scope: &'s TaskScope<'s, 'env>,
    senders: Vec<PortSender>,
    seq: &'s AtomicU64,
    seq_offset: MorselSeq,
    get_df: F,
    join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
) where
    F: Fn(u64) -> Fut + Copy + Send + 's,
    Fut: Future<Output = Option<DataFrame>> + Send + 's,
{
    let source_token = SourceToken::new();
    for mut send in senders {
        let source_token = source_token.clone();
        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            let wait_group = WaitGroup::default();
            loop {
                let seq = seq.fetch_add(1, Ordering::Relaxed);
                let Some(df) = get_df(seq).await else {
                    break;
                };

                let morsel_seq = MorselSeq::new(seq).offset_by(seq_offset);
                let mut morsel = Morsel::new_unregistered(df, morsel_seq, source_token.clone());
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
