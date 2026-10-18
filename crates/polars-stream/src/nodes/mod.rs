pub mod backward_fill;
pub mod callback_sink;
pub mod columnar_function;
#[cfg(feature = "cum_agg")]
pub mod cum_agg;
#[cfg(feature = "dynamic_group_by")]
pub mod dynamic_group_by;
pub mod dynamic_slice;
#[cfg(feature = "ewma")]
pub mod ewm;
pub mod filter;
pub mod forward_fill;
pub mod gather;
pub mod gather_every;
pub mod group_by;
pub mod in_memory_map;
pub mod in_memory_sink;
pub mod in_memory_source;
pub mod input_independent_select;
#[cfg(feature = "interpolate")]
pub mod interpolate;
pub mod io_sinks;
pub mod io_sources;
#[cfg(feature = "is_first_distinct")]
pub mod is_first_distinct;
pub mod is_sorted;
pub mod joins;
pub mod map;
#[cfg(feature = "merge_sorted")]
pub mod merge_sorted;
pub mod multiplexer;
pub mod negative_slice;
pub mod ordered_union;
pub mod peak_minmax;
pub mod reduce;
pub mod repeat;
pub mod rle;
pub mod rle_id;
pub mod rolling_fixed_window;
#[cfg(feature = "dynamic_group_by")]
pub mod rolling_group_by;
pub mod scalar_window;
pub mod select;
pub mod shift;
pub mod simple_projection;
pub mod sort;
pub mod sorted_group_by;
pub mod sorted_unique;
pub mod streaming_slice;
#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-datetime",
    feature = "dtype-time"
))]
pub mod strptime_infer;
pub mod top_k;
pub mod unordered_union;
pub mod window;
pub mod with_row_index;
pub mod zip;

/// The imports you'll always need for implementing a ComputeNode.
mod compute_node_prelude {
    pub use polars_async::executor::{JoinHandle, TaskPriority, TaskScope};
    pub use polars_core::frame::DataFrame;
    pub use polars_error::PolarsResult;
    pub use polars_expr::state::ExecutionState;

    pub use super::{ComputeNode, NodeMemoryUsage};
    pub use crate::execute::StreamingExecutionState;
    pub use crate::graph::PortState;
    pub use crate::morsel::{Morsel, MorselSeq};
    pub use crate::pipe::{PortReceiver, PortSender, RecvPort, SendPort};
}

use compute_node_prelude::*;

use crate::execute::StreamingExecutionState;

/// How a node's memory use relates to its input size, given its current state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeMemoryUsage {
    /// Memory does not meaningfully grow with the input size, now or in a later
    /// state.
    Bounded,
    /// Memory may grow with the input size while data streams through the node.
    Unbounded,

    /// Stores nothing meaningful yet, but will be `Accumulating` in a later state.
    WillAccumulate,
    /// Stores (substantial parts of) its input while running. Each phase runs
    /// at most one such node, unless all such nodes remaining are sinks.
    Accumulating,
    /// Holds significant stored data that may be freed once this node runs to completion.
    HoldingUntilDone,
    /// Holds significant stored data that may be freed immediately while this node runs.
    Draining,
}

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

    /// The memory usage of this node in its current state.
    fn memory_usage(&self) -> NodeMemoryUsage {
        NodeMemoryUsage::Bounded
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
