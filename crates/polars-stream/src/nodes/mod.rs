pub mod filter;
pub mod group_by;
pub mod in_memory_map;
pub mod in_memory_sink;
pub mod in_memory_source;
pub mod input_independent_select;
pub mod io_sinks;
pub mod io_sources;
pub mod joins;
pub mod map;
#[cfg(feature = "merge_sorted")]
pub mod merge_sorted;
pub mod multiplexer;
pub mod negative_slice;
pub mod ordered_union;
pub mod reduce;
pub mod select;
pub mod simple_projection;
pub mod streaming_slice;
pub mod with_row_index;
pub mod zip;

/// The imports you'll always need for implementing a ComputeNode.
mod compute_node_prelude {
    pub use polars_core::frame::DataFrame;
    pub use polars_error::PolarsResult;
    pub use polars_expr::state::ExecutionState;

    pub use super::ComputeNode;
    pub use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
    pub use crate::execute::StreamingExecutionState;
    pub use crate::graph::PortState;
    pub use crate::morsel::{Morsel, MorselSeq};
    pub use crate::pipe::{RecvPort, SendPort};
}

use compute_node_prelude::*;

use self::io_sources::PhaseOutcomeToken;
use crate::async_primitives::wait_group::WaitToken;
use crate::execute::StreamingExecutionState;

pub trait ComputeNode: Send {
    /// The name of this node.
    fn name(&self) -> &str;

    /// Update the state of this node given the state of our input and output
    /// ports. May be called multiple times until fully resolved for each
    /// execution phase.
    ///
    /// For each input pipe `recv` will contain a respective state of the
    /// send port that pipe is connected to when called, and it is expected when
    /// `update_state` returns it contains your computed receive port state.
    ///
    /// Similarly, for each output pipe `send` will contain the respective
    /// state of the input port that pipe is connected to when called, and you
    /// must update it to contain the desired state of your output port.
    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()>;

    /// If this node (in its current state) is a pipeline blocker, and whether
    /// this is memory intensive or not.
    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        false
    }

    /// Spawn the tasks that this compute node needs to receive input(s),
    /// process it and send to its output(s). Called once per execution phase.
    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );

    /// Called once after the last execution phase to extract output from
    /// in-memory nodes.
    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }
}

/// The outcome of a phase in a task.
///
/// This indicates whether a task finished (and does not need to be started again) or has stopped
/// prematurely. When this is dropped without calling `stop`, it is assumed that the task is
/// finished (most likely because it errored).
pub struct PhaseOutcome {
    // This is used to see when phase is finished.
    #[expect(unused)]
    consume_token: WaitToken,

    outcome_token: PhaseOutcomeToken,
}

impl PhaseOutcome {
    pub fn new_shared_wait(consume_token: WaitToken) -> (PhaseOutcomeToken, Self) {
        let outcome_token = PhaseOutcomeToken::new();
        (
            outcome_token.clone(),
            Self {
                consume_token,
                outcome_token,
            },
        )
    }

    /// Phase ended before the task finished and needs to be called again.
    pub fn stopped(self) {
        self.outcome_token.stop();
    }
}
