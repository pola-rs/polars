use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::async_executor;
use crate::async_primitives::pipe::{pipe, Receiver, Sender};
use crate::graph::{Graph, GraphNodeKey};
use crate::morsel::Morsel;

pub fn execute_graph(
    graph: &mut Graph,
) -> PolarsResult<SparseSecondaryMap<GraphNodeKey, DataFrame>> {
    // Get the number of threads from the rayon thread-pool as that respects our config.
    let num_pipes = POOL.current_num_threads();
    async_executor::set_num_threads(num_pipes);

    // Construct pipes.
    let mut physical_senders = SecondaryMap::new();
    let mut physical_receivers = SecondaryMap::new();

    // For morsel-driven parallelism we create N independent pipelines, where N is the number of threads.
    // The first step is to create N physical pipes for every logical pipe in the graph.
    for pipe_key in graph.pipes.keys() {
        let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
            (0..num_pipes).map(|_| pipe()).unzip();

        physical_senders.insert(pipe_key, senders);
        physical_receivers.insert(pipe_key, receivers);
    }

    let execution_state = ExecutionState::default();
    async_executor::task_scope(|scope| {
        // Initialize tasks.
        // This traverses the graph in arbitrary order. The order does not matter as the tasks will
        // simply wait for the input from their pipes until that input is ready.
        let mut join_handles = Vec::new();
        for node in graph.nodes.values_mut() {
            node.compute.initialize(num_pipes);

            let mut phys_inputs: Vec<_> = node
                .inputs
                .iter()
                .map(|i| physical_receivers.remove(*i).unwrap())
                .collect();
            let mut phys_outputs: Vec<_> = node
                .outputs
                .iter()
                .map(|o| physical_senders.remove(*o).unwrap())
                .collect();
            for pipeline in 0..num_pipes {
                let phys_input = phys_inputs.iter_mut().map(|i| i.pop().unwrap()).collect();
                let phys_output = phys_outputs.iter_mut().map(|o| o.pop().unwrap()).collect();
                join_handles.push(node.compute.spawn(
                    scope,
                    pipeline,
                    phys_input,
                    phys_output,
                    &execution_state,
                ));
            }
        }

        // Wait until all tasks are done.
        polars_io::pl_async::get_runtime().block_on(async move {
            for handle in join_handles {
                handle.await?;
            }
            PolarsResult::Ok(())
        })
    })?;

    // Finalize computation and get any in-memory results.
    let mut out = SparseSecondaryMap::new();
    for (node_key, node) in graph.nodes.iter_mut() {
        if let Some(df) = node.compute.finalize()? {
            out.insert(node_key, df);
        }
    }
    Ok(out)
}
