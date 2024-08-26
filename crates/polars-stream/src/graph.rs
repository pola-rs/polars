use polars_error::PolarsResult;
use slotmap::{SecondaryMap, SlotMap};

use crate::nodes::ComputeNode;

slotmap::new_key_type! {
    pub struct GraphNodeKey;
    pub struct LogicalPipeKey;
}

/// Represents the compute graph.
///
/// The `nodes` perform computation and the `pipes` form the connections between nodes
/// that data is sent through.
#[derive(Default)]
pub struct Graph {
    pub nodes: SlotMap<GraphNodeKey, GraphNode>,
    pub pipes: SlotMap<LogicalPipeKey, LogicalPipe>,
}

impl Graph {
    /// Allocate the needed `capacity` for the `Graph`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: SlotMap::with_capacity_and_key(capacity),
            pipes: SlotMap::with_capacity_and_key(capacity),
        }
    }

    /// Add a new `GraphNode` to the `Graph` and connect the inputs and outputs
    /// to their respective `LogicalPipe`s.
    pub fn add_node<N: ComputeNode + 'static>(
        &mut self,
        node: N,
        inputs: impl IntoIterator<Item = GraphNodeKey>,
    ) -> GraphNodeKey {
        // Add the GraphNode.
        let node_key = self.nodes.insert(GraphNode {
            compute: Box::new(node),
            inputs: Vec::new(),
            outputs: Vec::new(),
        });

        // Create and add pipes that connect input to output.
        for (recv_port, sender) in inputs.into_iter().enumerate() {
            let send_port = self.nodes[sender].outputs.len();
            let pipe = LogicalPipe {
                sender,
                send_port,
                send_state: PortState::Blocked,
                receiver: node_key,
                recv_port,
                recv_state: PortState::Blocked,
            };

            // Add the pipe.
            let pipe_key = self.pipes.insert(pipe);

            // And connect input to output.
            self.nodes[node_key].inputs.push(pipe_key);
            self.nodes[sender].outputs.push(pipe_key);
        }

        node_key
    }

    /// Updates all the nodes' states until a fixed point is reached.
    pub fn update_all_states(&mut self) -> PolarsResult<()> {
        let mut to_update: Vec<_> = self.nodes.keys().collect();
        let mut scheduled_for_update: SecondaryMap<GraphNodeKey, ()> =
            self.nodes.keys().map(|k| (k, ())).collect();

        let verbose = std::env::var("POLARS_VERBOSE_STATE_UPDATE").as_deref() == Ok("1");

        let mut recv_state = Vec::new();
        let mut send_state = Vec::new();
        while let Some(node_key) = to_update.pop() {
            scheduled_for_update.remove(node_key);
            let node = &mut self.nodes[node_key];

            // Get the states of nodes this node is connected to.
            recv_state.clear();
            send_state.clear();
            recv_state.extend(node.inputs.iter().map(|i| self.pipes[*i].send_state));
            send_state.extend(node.outputs.iter().map(|o| self.pipes[*o].recv_state));

            // Compute the new state of this node given its environment.
            if verbose {
                eprintln!(
                    "updating {}, before: {recv_state:?} {send_state:?}",
                    node.compute.name()
                );
            }
            node.compute
                .update_state(&mut recv_state, &mut send_state)?;
            if verbose {
                eprintln!(
                    "updating {}, after: {recv_state:?} {send_state:?}",
                    node.compute.name()
                );
            }

            // Propagate information.
            for (input, state) in node.inputs.iter().zip(recv_state.iter()) {
                let pipe = &mut self.pipes[*input];
                if pipe.recv_state != *state {
                    assert!(pipe.recv_state != PortState::Done, "implementation error: state transition from Done to Blocked/Ready attempted");
                    pipe.recv_state = *state;
                    if scheduled_for_update.insert(pipe.sender, ()).is_none() {
                        to_update.push(pipe.sender);
                    }
                }
            }

            for (output, state) in node.outputs.iter().zip(send_state.iter()) {
                let pipe = &mut self.pipes[*output];
                if pipe.send_state != *state {
                    assert!(pipe.send_state != PortState::Done, "implementation error: state transition from Done to Blocked/Ready attempted");
                    pipe.send_state = *state;
                    if scheduled_for_update.insert(pipe.receiver, ()).is_none() {
                        to_update.push(pipe.receiver);
                    }
                }
            }
        }
        Ok(())
    }
}

/// A node in the graph represents a computation performed on the stream of morsels
/// that flow through it.
pub struct GraphNode {
    pub compute: Box<dyn ComputeNode>,
    pub inputs: Vec<LogicalPipeKey>,
    pub outputs: Vec<LogicalPipeKey>,
}

/// A pipe sends data between nodes.
#[allow(unused)] // TODO: remove.
pub struct LogicalPipe {
    // Node that we send data to.
    pub sender: GraphNodeKey,
    // Output location:
    // graph[x].output[i].send_port == i
    send_port: usize,
    pub send_state: PortState,

    // Node that we receive data from.
    pub receiver: GraphNodeKey,
    // Input location:
    // graph[x].inputs[i].recv_port == i
    recv_port: usize,
    pub recv_state: PortState,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum PortState {
    Blocked,
    Ready,
    Done,
}
