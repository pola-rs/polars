mod edge_provider;
mod ir_graph;
mod ir_node_key;

pub use edge_provider::IRTraversalGraphEdgeProvider;
pub use ir_graph::{IRNodeEdgeKeys, build_ir_traversal_graph, unpack_edges_mut};
pub use ir_node_key::IRNodeKey;
