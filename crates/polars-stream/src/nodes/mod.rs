use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use crate::async_executor::{JoinHandle, TaskScope};
use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

pub mod filter;
pub mod in_memory_sink;
pub mod in_memory_source;
pub mod simple_projection;

pub trait ComputeNode {
    /// Initialize for processing using the given amount of pipelines.
    fn initialize(&mut self, _num_pipelines: usize) {}

    /// Spawn a task that should receive input(s), process it and send to its
    /// output(s). Called once for each pipeline.
    fn spawn<'env, 's>(
        &'env self,
        scope: &'s TaskScope<'s, 'env>,
        _pipeline: usize,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        state: &'s ExecutionState,
    ) -> JoinHandle<PolarsResult<()>>;

    /// Called after this computation is complete.
    fn finalize(&mut self) -> PolarsResult<Option<DataFrame>> {
        Ok(None)
    }
}
