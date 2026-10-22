use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_async::primitives::wait_group::WaitGroup;
use polars_core::schema::SchemaRef;
use polars_ooc::SpillFrame;

use super::compute_node_prelude::*;
use crate::morsel::{MorselSeq, SourceToken, get_ideal_morsel_size};

/// Sends the rows of frames that can be spilled. A frame is only loaded while it is sliced into
/// morsels, so the frames stay spillable while the node runs.
pub struct SpillFrameSourceNode {
    /// Groups of frames, sent in order.
    groups: Vec<Arc<[SpillFrame]>>,
    schema: SchemaRef,
    /// The group, frame, offset and length of every morsel.
    morsels: Vec<(usize, usize, usize, usize)>,
    next_morsel: AtomicUsize,
    started: bool,
}

impl SpillFrameSourceNode {
    pub fn new(groups: Vec<Arc<[SpillFrame]>>, schema: SchemaRef) -> Self {
        Self {
            groups,
            schema,
            morsels: Vec::new(),
            next_morsel: AtomicUsize::new(0),
            started: false,
        }
    }

    /// The number of morsels to send. An empty input still sends one empty morsel.
    fn num_morsels(&self) -> usize {
        self.morsels.len().max(1)
    }
}

impl ComputeNode for SpillFrameSourceNode {
    fn name(&self) -> &str {
        "spill-frame-source"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert!(send.len() == 1);

        if !self.started {
            self.started = true;
            let frames = || self.groups.iter().flat_map(|group| group.iter());
            let len: usize = frames().map(|sf| sf.height()).sum();
            let ideal_morsel_count = (len / get_ideal_morsel_size()).max(1);
            let morsel_count = ideal_morsel_count.next_multiple_of(state.num_pipelines);
            let morsel_size = len.div_ceil(morsel_count).max(1);
            for (g, group) in self.groups.iter().enumerate() {
                for (f, sf) in group.iter().enumerate() {
                    for offset in (0..sf.height()).step_by(morsel_size) {
                        let len = morsel_size.min(sf.height() - offset);
                        self.morsels.push((g, f, offset, len));
                    }
                }
            }
        }

        let exhausted = self.next_morsel.load(Ordering::Relaxed) >= self.num_morsels();
        if send[0] == PortState::Done || exhausted {
            send[0] = PortState::Done;
            self.groups = Vec::new();
            self.morsels = Vec::new();
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.is_empty() && send_ports.len() == 1);
        let senders = send_ports[0].take().unwrap().parallel();

        let source_token = SourceToken::new();
        for mut send in senders {
            let slf = &*self;
            let source_token = source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                let wait_group = WaitGroup::default();
                loop {
                    let i = slf.next_morsel.fetch_add(1, Ordering::Relaxed);
                    if i >= slf.num_morsels() {
                        break;
                    }
                    let df = match slf.morsels.get(i) {
                        Some(&(g, f, offset, len)) => {
                            slf.groups[g][f].get().await.slice(offset as i64, len)
                        },
                        None => DataFrame::empty_with_schema(&slf.schema),
                    };

                    let morsel_seq = MorselSeq::new(i as u64);
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
}
