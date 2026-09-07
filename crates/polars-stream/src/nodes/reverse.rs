use std::collections::VecDeque;
use std::sync::Arc;

use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_ooc::{MostRecentSpillContext, ParameterFreeSpillContext, SpillFrame};

use super::compute_node_prelude::*;
use crate::nodes::in_memory_source::InMemorySourceNode;

// A lot of the code in this module is similar to that in `negative_slice`.

/// The buffer is put in the in the enum rather than as state in the node itself to make illegal
/// states impossible to represent: `Done` cannot accidentally leak data that way.
enum ReverseState {
    Buffering(VecDeque<SpillFrame>),
    Emitting {
        frames: VecDeque<SpillFrame>,
        seq: MorselSeq,
    },
    Done,
}

pub struct ReverseNode {
    state: ReverseState,
    spill_ctx: MostRecentSpillContext,
}

impl ReverseNode {
    pub fn new() -> ReverseNode {
        ReverseNode {
            state: ReverseState::Buffering(VecDeque::new()),
            spill_ctx: MostRecentSpillContext::new("reverse".into()),
        }
    }
}

impl ComputeNode for ReverseNode {
    fn name(&self) -> &str {
        "reverse"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        // Stop streaming if downstream says it is done.
        if send[0] == PortState::Done {
            self.state = ReverseState::Done;
        }
        if matches!(
            (recv[0], &self.state),
            (PortState::Done, ReverseState::Buffering(_))
        ) {
            // Just received the last morsels.
            // Transition to becoming a source if there is anything to feed.
            let ReverseState::Buffering(frames) =
                core::mem::replace(&mut self.state, ReverseState::Done)
            else {
                unreachable!()
            };
            if !frames.is_empty() {
                self.state = ReverseState::Emitting {
                    frames,
                    seq: MorselSeq::default(),
                }
            }
        }

        match &mut self.state {
            ReverseState::Buffering(_) => {
                recv[0] = PortState::Ready;
                send[0] = PortState::Blocked;
            },
            ReverseState::Emitting { frames, seq: _ } => {
                recv[0] = PortState::Done;
                // InMemorySource has implemented a hack for compatibility with
                // nodes downstream that require at least one input.
                // Do we need to copy this?
                send[0] = if frames.is_empty() {
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
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        todo!()
    }
}
