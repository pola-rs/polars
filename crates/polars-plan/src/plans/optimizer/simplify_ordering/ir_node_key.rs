use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::IR;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
enum Inner {
    Node(Node),
    CacheId(UniqueId),
}

/// IR node key that uses the cache ID for cache nodes.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct IRNodeKey(Inner);

impl IRNodeKey {
    pub fn new(ir_node: Node, ir_arena: &Arena<IR>) -> Self {
        Self(match ir_arena.get(ir_node) {
            IR::Cache { id, .. } => Inner::CacheId(*id),
            _ => Inner::Node(ir_node),
        })
    }
}
