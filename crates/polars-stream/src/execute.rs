use polars_core::frame::DataFrame;
use polars_core::POOL;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_utils::aliases::PlHashSet;
use slotmap::{SecondaryMap, SparseSecondaryMap};

use crate::async_executor;
use crate::async_primitives::pipe::{pipe, Receiver, Sender};
use crate::graph::{Graph, GraphNodeKey, LogicalPipeKey, PortState};
use crate::morsel::Morsel;

/// Finds all runnable pipeline blockers in the graph, that is, nodes which:
///  - Only have blocked output ports.
///  - Have at least one ready input port connected to a ready output port.
fn find_runnable_pipeline_blockers(graph: &Graph) -> Vec<GraphNodeKey> {
    let mut blockers = Vec::new();
    for (node_key, node) in graph.nodes.iter() {
        // TODO: how does the multiplexer fit into this?
        let only_has_blocked_outputs = node
            .outputs
            .iter()
            .all(|o| graph.pipes[*o].send_state == PortState::Blocked);
        if !only_has_blocked_outputs {
            continue;
        }

        let has_input_ready = node.inputs.iter().any(|i| {
            graph.pipes[*i].send_state == PortState::Ready
                && graph.pipes[*i].recv_state == PortState::Ready
        });
        if has_input_ready {
            blockers.push(node_key);
        }
    }
    blockers
}

/// Given a set of nodes expand this set with all nodes which are inputs to the
/// set and whose connecting pipe is ready on both sides, recursively.
///
/// Returns the set of nodes as well as the pipes connecting them.
fn expand_ready_subgraph(
    graph: &Graph,
    mut nodes: Vec<GraphNodeKey>,
) -> (PlHashSet<GraphNodeKey>, Vec<LogicalPipeKey>) {
    let mut in_subgraph: PlHashSet<GraphNodeKey> = nodes.iter().copied().collect();
    let mut pipes = Vec::with_capacity(nodes.len());
    while let Some(node_key) = nodes.pop() {
        let node = &graph.nodes[node_key];
        for input_pipe_key in &node.inputs {
            let input_pipe = &graph.pipes[*input_pipe_key];
            if input_pipe.send_state == PortState::Ready
                && input_pipe.recv_state == PortState::Ready
            {
                pipes.push(*input_pipe_key);
                if in_subgraph.insert(input_pipe.sender) {
                    nodes.push(input_pipe.sender);
                }
            }
        }
    }

    (in_subgraph, pipes)
}

/// Finds a part of the graph which we can run.
fn find_runnable_subgraph(graph: &mut Graph) -> (PlHashSet<GraphNodeKey>, Vec<LogicalPipeKey>) {
    // Find pipeline blockers, choose a subset with at most one memory intensive
    // pipeline blocker, and return the subgraph needed to feed them.
    let blockers = find_runnable_pipeline_blockers(graph);
    let (mut expensive, cheap): (Vec<_>, Vec<_>) = blockers.into_iter().partition(|b| {
        graph.nodes[*b]
            .compute
            .is_memory_intensive_pipeline_blocker()
    });

    // TODO: choose which expensive pipeline blocker to run more intelligently.
    expensive.sort_by_key(|node_key| {
        // Prefer to run nodes whose outputs are ready to be consumed.
        let outputs_ready_to_receive = graph.nodes[*node_key]
            .outputs
            .iter()
            .filter(|o| graph.pipes[**o].recv_state == PortState::Ready)
            .count();
        outputs_ready_to_receive
    });

    let mut to_run = cheap;
    if let Some(node) = expensive.pop() {
        to_run.push(node);
    }
    expand_ready_subgraph(graph, to_run)
}

/// Runs the given subgraph. Assumes the set of pipes is correct for the subgraph.
fn run_subgraph(
    graph: &mut Graph,
    nodes: &PlHashSet<GraphNodeKey>,
    pipes: &[LogicalPipeKey],
    num_pipelines: usize,
) -> PolarsResult<()> {
    // Construct pipes.
    let mut physical_senders = SecondaryMap::new();
    let mut physical_receivers = SecondaryMap::new();

    // For morsel-driven parallelism we create N independent pipelines, where N is the number of threads.
    // The first step is to create N physical pipes for every logical pipe in the graph.
    for pipe_key in pipes.iter().copied() {
        let (senders, receivers): (Vec<Sender<Morsel>>, Vec<Receiver<Morsel>>) =
            (0..num_pipelines).map(|_| pipe()).unzip();

        physical_senders.insert(pipe_key, senders);
        physical_receivers.insert(pipe_key, receivers);
    }

    let execution_state = ExecutionState::default();
    async_executor::task_scope(|scope| {
        // Initialize tasks.
        // This traverses the graph in arbitrary order. The order does not matter as the tasks will
        // simply wait for the input from their pipes until that input is ready.
        let mut join_handles = Vec::new();
        let mut phys_recv = Vec::new();
        let mut phys_send = Vec::new();
        for (node_key, node) in graph.nodes.iter_mut() {
            // We can't directly loop over nodes because we need iter_mut to get
            // multiple mutable references without the compiler complaining about
            // borrowing graph.nodes while it was borrowed previous iteration.
            if !nodes.contains(&node_key) {
                continue;
            }

            // Scatter inputs/outputs per pipeline.
            let num_inputs = node.inputs.len();
            let num_outputs = node.outputs.len();
            phys_recv.resize_with(num_inputs * num_pipelines, || None);
            for (input_idx, input) in node.inputs.iter().copied().enumerate() {
                if let Some(receivers) = physical_receivers.remove(input) {
                    for (recv_idx, recv) in receivers.into_iter().enumerate() {
                        phys_recv[recv_idx * num_inputs + input_idx] = Some(recv);
                    }
                }
            }

            phys_send.resize_with(num_outputs * num_pipelines, || None);
            for (output_idx, output) in node.outputs.iter().copied().enumerate() {
                if let Some(senders) = physical_senders.remove(output) {
                    for (send_idx, send) in senders.into_iter().enumerate() {
                        phys_send[send_idx * num_outputs + output_idx] = Some(send);
                    }
                }
            }

            // Spawn the global task, if any.
            if let Some(handle) = node.compute.spawn_global(scope, &execution_state) {
                join_handles.push(handle);
            }

            // Spawn a task per pipeline.
            for pipeline in 0..num_pipelines {
                join_handles.push(node.compute.spawn(
                    scope,
                    pipeline,
                    &mut phys_recv[num_inputs * pipeline..num_inputs * (pipeline + 1)],
                    &mut phys_send[num_outputs * pipeline..num_outputs * (pipeline + 1)],
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

    Ok(())
}

pub fn execute_graph(
    graph: &mut Graph,
) -> PolarsResult<SparseSecondaryMap<GraphNodeKey, DataFrame>> {
    // Get the number of threads from the rayon thread-pool as that respects our config.
    let num_pipelines = POOL.current_num_threads();
    async_executor::set_num_threads(num_pipelines);

    for node in graph.nodes.values_mut() {
        node.compute.initialize(num_pipelines);
    }

    loop {
        if polars_core::config::verbose() {
            eprintln!("polars-stream: updating graph state");
        }
        graph.update_all_states();
        let (nodes, pipes) = find_runnable_subgraph(graph);
        if polars_core::config::verbose() {
            for node in &nodes {
                eprintln!(
                    "polars-stream: running {} in subgraph",
                    graph.nodes[*node].compute.name()
                );
            }
        }
        if nodes.is_empty() {
            break;
        }
        run_subgraph(graph, &nodes, &pipes, num_pipelines)?;
    }

    // Ensure everything is done.
    for pipe in graph.pipes.values() {
        assert!(pipe.send_state == PortState::Done && pipe.recv_state == PortState::Done);
    }

    // Extract output from in-memory nodes.
    let mut out = SparseSecondaryMap::new();
    for (node_key, node) in graph.nodes.iter_mut() {
        if let Some(df) = node.compute.get_output()? {
            out.insert(node_key, df);
        }
    }

    Ok(out)
}
