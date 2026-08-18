//! Defines different visitor patterns and for any tree.

use arrow::legacy::error::PolarsResult;
mod expr;
mod lp;
mod visitors;

pub use expr::*;
pub use lp::*;
pub use visitors::*;

/// Controls how the [`TreeWalker`] recursion should proceed for [`TreeWalker::visit`].
#[derive(Debug)]
pub enum VisitRecursion {
    /// Visit this node's children, then call [`Visitor::post_visit`] for this node.
    Continue,
    /// Skip this node's children and do not call [`Visitor::post_visit`] for this node.
    ///
    /// Traversal continues with the next sibling or ancestor.
    Skip,
    /// Stop the entire traversal immediately.
    ///
    /// [`Visitor::post_visit`] is not called for this node or for any active ancestor.
    Stop,
}

/// Controls how the [`TreeWalker`] recursion should proceed for [`TreeWalker::rewrite`].
#[derive(Debug)]
pub enum RewriteRecursion {
    /// Continue the visit to this node and children.
    MutateAndContinue,
    /// Don't mutate this node, continue visiting the children
    NoMutateAndContinue,
    /// Stop and return.
    /// This doesn't visit the children
    Stop,
    /// Call `op` immediately and return
    /// This doesn't visit the children
    MutateAndStop,
}
