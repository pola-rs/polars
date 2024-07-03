//! Defines different visitor patterns and for any tree.

use arrow::legacy::error::PolarsResult;
mod expr;
#[cfg(feature = "cse")]
mod hash;
mod lp;
mod visitors;

pub use expr::*;
pub use lp::*;
pub use visitors::*;

/// Controls how the [`TreeWalker`] recursion should proceed for [`TreeWalker::visit`].
#[derive(Debug)]
pub enum VisitRecursion {
    /// Continue the visit to this node tree.
    Continue,
    /// Keep recursive but skip applying op on the children
    Skip,
    /// Stop the visit to this node tree.
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
