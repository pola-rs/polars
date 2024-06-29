use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::graph::PortState;
use crate::morsel::Morsel;

pub mod filter;
pub mod in_memory_map;
pub mod in_memory_sink;
pub mod in_memory_source;
pub mod map;
pub mod select;
pub mod simple_projection;

pub trait ComputeNode: Send + Sync {
    fn name(&self) -> &'static str;

    /// Update the state of this node given the state of our input and output
    /// ports. May be called multiple times until fully resolved for each
    /// execution cycle.
    ///
    /// For each input pipe `recv` will contain a respective state of the
    /// send port that pipe is connected to when called, and it is expected when
    /// `update_state` returns it contains your computed receive port state.
    ///
    /// Similarly, for each output pipe `send` will contain the respective
    /// state of the input port that pipe is connected to when called, and you
    /// must update it to contain the desired state of your output port.
    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]);

    /// If this node (in its current state) is a pipeline blocker, and whether
    /// this is memory intensive or not.
    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        false
    }

    /// Initialize for processing using the given amount of pipelines.
    fn initialize(&mut self, _num_pipelines: usize) {}

    /// Spawn a task that should receive input(s), process it and send to its
    /// output(s). Called once for each pipeline.
    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: &mut [Option<Receiver<Morsel>>],
        send: &mut [Option<Sender<Morsel>>],
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>>;

    /// Called after this computation is complete.
    fn finalize(&mut self) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }
}
