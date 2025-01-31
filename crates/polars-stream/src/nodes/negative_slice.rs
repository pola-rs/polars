use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::compute_node_prelude::*;
use crate::nodes::in_memory_source::InMemorySourceNode;

/// A node that will pass-through up to length rows, starting at start_offset.
/// Since start_offset must be non-negative this can be done in a streaming
/// manner.
enum NegativeSliceState {
    Buffering(Buffer),
    Source(InMemorySourceNode),
    Done,
}

#[derive(Default)]
struct Buffer {
    frames: VecDeque<DataFrame>,
    total_len: usize,
}

pub struct NegativeSliceNode {
    state: NegativeSliceState,
    slice_offset: i64,
    length: usize,
    num_pipelines: usize,
}

impl NegativeSliceNode {
    pub fn new(slice_offset: i64, length: usize) -> Self {
        assert!(slice_offset < 0);
        Self {
            state: NegativeSliceState::Buffering(Buffer::default()),
            slice_offset,
            length,
            num_pipelines: 0,
        }
    }
}

impl ComputeNode for NegativeSliceNode {
    fn name(&self) -> &str {
        "negative_slice"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        use NegativeSliceState::*;

        if send[0] == PortState::Done || self.length == 0 {
            self.state = Done;
        }

        if recv[0] == PortState::Done {
            if let Buffering(buffer) = &mut self.state {
                // These offsets are relative to the start of buffer.
                let mut signed_start_offset = buffer.total_len as i64 + self.slice_offset;
                let signed_stop_offset =
                    signed_start_offset.saturating_add_unsigned(self.length as u64);

                // Trim the frames in the buffer to just those that are relevant.
                while buffer.total_len > 0
                    && signed_start_offset >= buffer.frames.front().unwrap().height() as i64
                {
                    let len = buffer.frames.pop_front().unwrap().height();
                    buffer.total_len -= len;
                    signed_start_offset -= len as i64;
                }

                while buffer.total_len as i64 - buffer.frames.back().unwrap().height() as i64
                    > signed_stop_offset
                {
                    buffer.total_len -= buffer.frames.pop_back().unwrap().height();
                }

                if buffer.total_len == 0 {
                    self.state = Done;
                } else {
                    let mut df = accumulate_dataframes_vertical_unchecked(buffer.frames.drain(..));
                    df = df.slice(signed_start_offset, self.length);
                    let mut node = InMemorySourceNode::new(Arc::new(df), MorselSeq::default());
                    node.initialize(self.num_pipelines);
                    self.state = Source(node);
                }
            }
        }

        match &mut self.state {
            Buffering(_) => {
                recv[0] = PortState::Ready;
                send[0] = PortState::Blocked;
            },
            Source(node) => {
                recv[0] = PortState::Done;
                node.update_state(&mut [], send)?;
            },
            Done => {
                recv[0] = PortState::Done;
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        match &mut self.state {
            NegativeSliceState::Buffering(buffer) => {
                let mut recv = recv_ports[0].take().unwrap().serial();
                assert!(send_ports[0].is_none());
                let max_buffer_needed = self.slice_offset.unsigned_abs() as usize;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        buffer.total_len += morsel.df().height();
                        buffer.frames.push_back(morsel.into_df());

                        if buffer.total_len - buffer.frames.front().unwrap().height()
                            >= max_buffer_needed
                        {
                            buffer.total_len -= buffer.frames.pop_front().unwrap().height();
                        }
                    }

                    Ok(())
                }));
            },
            NegativeSliceState::Source(in_memory_source_node) => {
                assert!(recv_ports[0].is_none());
                in_memory_source_node.spawn(scope, &mut [], send_ports, state, join_handles);
            },
            NegativeSliceState::Done => unreachable!(),
        }
    }
}
