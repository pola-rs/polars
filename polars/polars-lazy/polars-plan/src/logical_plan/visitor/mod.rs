//! Defines different visitor patterns and sort-orders for any tree.
//! See more on tree-traversal https://en.wikipedia.org/wiki/Tree_traversal

use polars_arrow::error::PolarsResult;
mod expr;
mod tree_node;

pub(crate) use expr::*;
pub(crate) use tree_node::*;

/// Controls how the [`TreeNode`] recursion should proceed for [`TreeNode::visit`].
#[derive(Debug)]
pub enum VisitRecursion {
    /// Continue the visit to this node tree.
    Continue,
    /// Keep recursive but skip applying op on the children
    Skip,
    /// Stop the visit to this node tree.
    Stop,
}

// #[derive(Debug)]
// pub enum RewriteRecursion {
//     /// Continue rewrite this node tree.
//     Continue,
//     /// Call 'op' immediately and return.
//     Mutate,
//     /// Do not rewrite the children of this node.
//     Stop,
//     /// Keep recursive but skip apply op on this node
//     Skip,
// }
//
// trait TreeNodeRewriter {
//     type Node;
//
//     /// invoked before any children of node are visited.
//     fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<RewriteRecursion> {
//         Ok(RewriteRecursion::Continue)
//     }
//
//     /// invoked after all children of `node` are visited. It returns a potentially
//     /// modified node.
//     fn mutate(&mut self, node: Self::Node) -> PolarsResult<Self::Node>;
// }
