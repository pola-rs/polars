pub mod filter;
pub mod in_memory_map;
pub mod in_memory_sink;
pub mod in_memory_source;
pub mod map;
pub mod multiplexer;
pub mod ordered_union;
pub mod parquet_source;
pub mod reduce;
pub mod select;
pub mod simple_projection;
pub mod streaming_slice;
pub mod zip;

/// The imports you'll always need for implementing a ComputeNode.
mod compute_node_prelude {
    pub use polars_core::frame::DataFrame;
    pub use polars_error::PolarsResult;
    pub use polars_expr::state::ExecutionState;

    pub use super::ComputeNode;
    pub use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
    pub use crate::graph::PortState;
    pub use crate::morsel::{Morsel, MorselSeq};
    pub use crate::pipe::{RecvPort, SendPort};
}

use compute_node_prelude::*;

pub trait ComputeNode: Send {
    /// The name of this node.
    fn name(&self) -> &str;

    /// Called once before the first execution phase to indicate with how many
    /// pipelines we will execute the graph.
    fn initialize(&mut self, _num_pipelines: usize) {}

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
    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()>;

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
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    );

    /// Called once after the last execution phase to extract output from
    /// in-memory nodes.
    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }
}
