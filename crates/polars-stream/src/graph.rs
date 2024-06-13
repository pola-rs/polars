use slotmap::SlotMap;

use crate::nodes::ComputeNode;

slotmap::new_key_type! {
    pub struct GraphNodeKey;
    pub struct LogicalPipeKey;
}

#[derive(Default)]
pub struct Graph {
    nodes: SlotMap<GraphNodeKey, GraphNode>,
    pipes: SlotMap<LogicalPipeKey, LogicalPipe>,
}

impl Graph {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: SlotMap::with_capacity_and_key(capacity),
            pipes: SlotMap::with_capacity_and_key(capacity),
        }
    }

    pub fn add_node<N: ComputeNode + 'static>(
        &mut self,
        node: N,
        inputs: impl IntoIterator<Item = GraphNodeKey>,
    ) -> GraphNodeKey {
        let node_key = self.nodes.insert(GraphNode {
            compute: Box::new(node),
            inputs: Vec::new(),
            outputs: Vec::new(),
        });

        for (recv_port, sender) in inputs.into_iter().enumerate() {
            let send_port = self.nodes[sender].outputs.len();
            let pipe = LogicalPipe {
                sender,
                send_port,
                receiver: node_key,
                recv_port,
            };

            let pipe_key = self.pipes.insert(pipe);
            self.nodes[node_key].inputs.push(pipe_key);
            self.nodes[sender].outputs.push(pipe_key);
        }

        node_key
    }
}

pub struct GraphNode {
    compute: Box<dyn ComputeNode>,
    inputs: Vec<LogicalPipeKey>,
    outputs: Vec<LogicalPipeKey>,
}

pub struct LogicalPipe {
    sender: GraphNodeKey,
    send_port: usize,

    receiver: GraphNodeKey,
    recv_port: usize,
}
