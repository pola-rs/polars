use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_core::schema::SchemaRef;
use polars_ooc::SpillFrame;

use super::compute_node_prelude::*;
use super::in_memory_source::{in_memory_morsel_size, spawn_in_memory_senders};

/// Sends the rows of frames that can be spilled. A frame is only pinned while it is sliced into a
/// morsel, so it can be spilled again while the node runs.
pub struct SpillFrameSourceNode {
    /// Groups of frames, sent in order.
    groups: Vec<Arc<[SpillFrame]>>,
    schema: SchemaRef,
    /// The group, frame, offset and length of every morsel.
    morsels: Vec<(usize, usize, usize, usize)>,
    next_morsel: AtomicU64,
    started: bool,
}

impl SpillFrameSourceNode {
    pub fn new(groups: Vec<Arc<[SpillFrame]>>, schema: SchemaRef) -> Self {
        Self {
            groups,
            schema,
            morsels: Vec::new(),
            next_morsel: AtomicU64::new(0),
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
            let morsel_size = in_memory_morsel_size(len, state.num_pipelines);
            for (g, group) in self.groups.iter().enumerate() {
                for (f, sf) in group.iter().enumerate() {
                    for offset in (0..sf.height()).step_by(morsel_size) {
                        let len = morsel_size.min(sf.height() - offset);
                        self.morsels.push((g, f, offset, len));
                    }
                }
            }
        }

        let exhausted = self.next_morsel.load(Ordering::Relaxed) as usize >= self.num_morsels();
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
        let slf = &*self;
        spawn_in_memory_senders(
            scope,
            senders,
            &slf.next_morsel,
            MorselSeq::default(),
            move |i| async move {
                let i = i as usize;
                if i >= slf.num_morsels() {
                    return None;
                }
                Some(match slf.morsels.get(i) {
                    Some(&(g, f, offset, len)) => {
                        slf.groups[g][f].get().await.slice(offset as i64, len)
                    },
                    None => DataFrame::empty_with_schema(&slf.schema),
                })
            },
            join_handles,
        );
    }
}
