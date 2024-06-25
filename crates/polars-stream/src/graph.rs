use slotmap::SlotMap;

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
                receiver: node_key,
                recv_port,
            };

            // Add the pipe.
            let pipe_key = self.pipes.insert(pipe);

            // And connect input to output.
            self.nodes[node_key].inputs.push(pipe_key);
            self.nodes[sender].outputs.push(pipe_key);
        }

        node_key
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
pub struct LogicalPipe {
    // Node that we send data to.
    sender: GraphNodeKey,
    // Output location:
    // graph[x].output[i].send_port == i
    send_port: usize,

    // Node that we receive data from.
    receiver: GraphNodeKey,
    // Input location:
    // graph[x].inputs[i].recv_port == i
    recv_port: usize,
}
