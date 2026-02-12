use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_ooc::mm;

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
    tokens: VecDeque<Token>,
    total_len: usize,
}

pub struct NegativeSliceNode {
    state: NegativeSliceState,
    slice_offset: i64,
    length: usize,
}

impl NegativeSliceNode {
    pub fn new(slice_offset: i64, length: usize) -> Self {
        assert!(slice_offset < 0);
        Self {
            state: NegativeSliceState::Buffering(Buffer::default()),
            slice_offset,
            length,
        }
    }
}

impl ComputeNode for NegativeSliceNode {
    fn name(&self) -> &str {
        "negative-slice"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        use NegativeSliceState::*;

        if send[0] == PortState::Done || self.length == 0 {
            if let Buffering(buffer) = &mut self.state {
                mm().drop_all_sync(buffer.tokens.drain(..))?;
            }
            self.state = Done;
        }

        if recv[0] == PortState::Done {
            if let Buffering(buffer) = &mut self.state {
                // These offsets are relative to the start of buffer.
                let mut signed_start_offset = buffer.total_len as i64 + self.slice_offset;
                let signed_stop_offset =
                    signed_start_offset.saturating_add_unsigned(self.length as u64);

                // Trim the tokens in the buffer to just those that are relevant.
                while buffer.total_len > 0
                    && signed_start_offset >= buffer.tokens.front().unwrap().height() as i64
                {
                    let token = buffer.tokens.pop_front().unwrap();
                    let len = token.height();
                    mm().drop_sync(token)?;
                    buffer.total_len -= len;
                    signed_start_offset -= len as i64;
                }

                while !buffer.tokens.is_empty()
                    && buffer.total_len as i64 - buffer.tokens.back().unwrap().height() as i64
                        > signed_stop_offset
                {
                    let token = buffer.tokens.pop_back().unwrap();
                    buffer.total_len -= token.height();
                    mm().drop_sync(token)?;
                }

                if buffer.total_len == 0 {
                    self.state = Done;
                } else {
                    let mut df = accumulate_dataframes_vertical_unchecked(
                        mm().take_all_sync(buffer.tokens.drain(..))?,
                    );
                    let clamped_start = signed_start_offset.max(0);
                    let len = (signed_stop_offset - clamped_start).max(0) as usize;
                    df = df.slice(clamped_start, len);
                    self.state =
                        Source(InMemorySourceNode::new(Arc::new(df), MorselSeq::default()));
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
                node.update_state(&mut [], send, state)?;
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
        state: &'s StreamingExecutionState,
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
                        buffer.tokens.push_back(mm().store(morsel.into_df()).await?);

                        if buffer.total_len - buffer.tokens.front().unwrap().height()
                            >= max_buffer_needed
                        {
                            let token = buffer.tokens.pop_front().unwrap();
                            buffer.total_len -= token.height();
                            mm().take(token).await?;
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
