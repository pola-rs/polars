use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;

use crate::async_primitives::pipe::{Receiver, Sender};
use crate::morsel::Morsel;

mod filter;
mod in_memory_source;

pub trait ComputeNode {
    /// Initialize for processing using the given amount of pipelines.
    fn initialize(&mut self, _num_pipelines: usize) { }

    /// A task that should receive input(s), process it and send to its output(s).
    /// Called once for each pipeline.
    async fn process(
        &self,
        recv: Vec<Receiver<Morsel>>,
        send: Vec<Sender<Morsel>>,
        state: &ExecutionState,
    ) -> PolarsResult<()>;
}
