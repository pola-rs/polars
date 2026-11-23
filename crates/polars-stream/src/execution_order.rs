use slotmap::SecondaryMap;

use crate::graph::{Graph, GraphNodeKey, PortState};
use crate::nodes::NodeMemoryUsage;

/// Chooses which of the given accumulating nodes to run, in order of preference:
/// 1. nodes whose outputs are ready to be consumed, so the result is not held;
/// 2. nodes that read from the most nodes holding data (see [`num_held_upstream`]), so that data
///    can be released;
/// 3. nodes with the most non-bounded nodes downstream (see [`num_non_bounded_downstream`]), so
///    a short branch does not hold its result while a long branch is processed;
/// 4. nodes with more outputs, so in-memory sinks (which have none) run last.
pub fn choose_accumulating_node(graph: &Graph, candidates: &[GraphNodeKey]) -> GraphNodeKey {
    let num_held = num_held_upstream(graph, candidates);
    let num_non_bounded = num_non_bounded_downstream(graph, candidates);
    *candidates
        .iter()
        .max_by_key(|node_key| {
            let node = &graph.nodes[**node_key];
            let num_outputs_ready_to_recv = node
                .outputs
                .iter()
                .filter(|o| graph.pipes[**o].recv_state == PortState::Ready)
                .count();
            (
                num_outputs_ready_to_recv,
                num_held[**node_key],
                num_non_bounded[**node_key],
                node.outputs.len(),
            )
        })
        .unwrap()
}

/// For each of the given nodes and the nodes feeding them through pipes ready on both sides, the
/// number of [`NodeMemoryUsage::HoldingUntilDone`] or [`NodeMemoryUsage::Draining`] nodes among
/// itself and its inputs through such pipes, recursively. A node reachable through several such
/// paths is counted once per path.
fn num_held_upstream(graph: &Graph, nodes: &[GraphNodeKey]) -> SecondaryMap<GraphNodeKey, usize> {
    let mut counts = SecondaryMap::new();
    let mut stack = nodes.to_vec();
    while let Some(&key) = stack.last() {
        if counts.contains_key(key) {
            stack.pop();
            continue;
        }
        let node = &graph.nodes[key];
        let mut n: usize = 0;
        let mut done = true;
        for pipe_key in &node.inputs {
            let pipe = &graph.pipes[*pipe_key];
            if pipe.send_state != PortState::Ready || pipe.recv_state != PortState::Ready {
                continue;
            }
            match counts.get(pipe.sender) {
                Some(c) => n = n.saturating_add(*c),
                None => {
                    stack.push(pipe.sender);
                    done = false;
                },
            }
        }
        if done {
            stack.pop();
            let held = matches!(
                node.compute.memory_usage(),
                NodeMemoryUsage::HoldingUntilDone | NodeMemoryUsage::Draining
            );
            counts.insert(key, n.saturating_add(held as usize));
        }
    }
    counts
}

/// For each of the given nodes and the nodes downstream of them, the largest number of nodes
/// that are not [`NodeMemoryUsage::Bounded`] on a path from that node (inclusive) to a node
/// without outputs.
fn num_non_bounded_downstream(
    graph: &Graph,
    nodes: &[GraphNodeKey],
) -> SecondaryMap<GraphNodeKey, usize> {
    let mut depths = SecondaryMap::new();
    let mut stack = nodes.to_vec();
    while let Some(&key) = stack.last() {
        if depths.contains_key(key) {
            stack.pop();
            continue;
        }
        let node = &graph.nodes[key];
        let mut d = 0;
        let mut done = true;
        for pipe_key in &node.outputs {
            let receiver = graph.pipes[*pipe_key].receiver;
            match depths.get(receiver) {
                Some(rd) => d = d.max(*rd),
                None => {
                    stack.push(receiver);
                    done = false;
                },
            }
        }
        if done {
            stack.pop();
            d += (node.compute.memory_usage() != NodeMemoryUsage::Bounded) as usize;
            depths.insert(key, d);
        }
    }
    depths
}
