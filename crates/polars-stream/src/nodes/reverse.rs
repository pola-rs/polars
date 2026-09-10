use std::collections::VecDeque;

use polars_async::primitives::wait_group::WaitGroup;
use polars_ooc::{LeastRecentSpillContext, ParameterFreeSpillContext, SpillFrame};

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;

// A lot of the code in this module is similar to that in `negative_slice`.

/// The buffer is put in the in the enum rather than as state in the node itself to make illegal
/// states impossible to represent: `Done` cannot accidentally leak data that way.
enum ReverseState {
    Buffering(Buffer),
    Emitting { buffer: Buffer, seq: MorselSeq },
    Done,
}

#[derive(Default)]
struct Buffer {
    frames: VecDeque<SpillFrame>,
    total_len: usize,
}

pub struct ReverseNode {
    state: ReverseState,
    /// Frames are emitted back-to-front, so it's best to spill the ones we received first
    spill_ctx: LeastRecentSpillContext,
}

impl ReverseNode {
    pub fn new() -> ReverseNode {
        ReverseNode {
            state: ReverseState::Buffering(Buffer::default()),
            spill_ctx: LeastRecentSpillContext::new("reverse".into()),
        }
    }
}

impl ComputeNode for ReverseNode {
    fn name(&self) -> &str {
        "reverse"
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        match &self.state {
            ReverseState::Buffering(..) => true,
            ReverseState::Emitting { .. } => false,
            ReverseState::Done => false,
        }
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        // Stop streaming if downstream says it is done.
        if send[0] == PortState::Done {
            self.state = ReverseState::Done;
        }

        if recv[0] == PortState::Done {
            if let ReverseState::Buffering(buffer) = &mut self.state {
                // Stop buffering and become a source.
                // If it is empty it is handled further down.
                self.state = ReverseState::Emitting {
                    buffer: core::mem::take(buffer),
                    seq: MorselSeq::default(),
                };
            }
        }

        match &mut self.state {
            ReverseState::Buffering(_) => {
                recv[0] = PortState::Ready;
                send[0] = PortState::Blocked;
            },
            ReverseState::Emitting { buffer, seq: _ } => {
                recv[0] = PortState::Done;
                // InMemorySource has implemented a hack for compatibility with
                // nodes downstream that require at least one input.
                // This is not needed here.

                send[0] = if buffer.total_len == 0 {
                    PortState::Done
                } else {
                    PortState::Ready
                }
            },
            ReverseState::Done => {
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
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);
        // Very similar to [super::negative_slice::NegativeSliceNode].
        match &mut self.state {
            ReverseState::Buffering(buffer) => {
                let mut recv = recv_ports[0].take().unwrap().serial();
                assert!(send_ports[0].is_none());
                let spill_ctx = self.spill_ctx.clone();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        buffer.total_len += morsel.height();
                        let sf = morsel.into_sf();
                        spill_ctx.register(&sf).await;
                        buffer.frames.push_back(sf);
                    }
                    Ok(())
                }));
            },
            ReverseState::Emitting { buffer, seq } => {
                assert!(recv_ports[0].is_none());
                let mut sender = send_ports[0].take().unwrap().serial();
                join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                    let source_token = SourceToken::new();
                    let wait_group = WaitGroup::default();
                    while let Some(sf) = buffer.frames.pop_back() {
                        buffer.total_len -= sf.height();
                        let df = sf.into_df().await.reverse();
                        let mut morsel = Morsel::new_unregistered(df, *seq, source_token.clone());
                        *seq = seq.successor();
                        morsel.set_consume_token(wait_group.token());
                        if sender.send(morsel).await.is_err() {
                            break;
                        }
                        wait_group.wait().await;
                        if source_token.stop_requested() {
                            break;
                        }
                    }
                    Ok(())
                }));
            },
            ReverseState::Done => unreachable!(),
        }
    }
}
