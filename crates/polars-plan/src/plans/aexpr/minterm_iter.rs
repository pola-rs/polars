use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::{AExpr, Operator};

/// An iterator over all the minterms in a boolean expression boolean.
///
/// In other words, all the terms that can `AND` together to form this expression.
///
/// # Example
///
/// ```
/// a & (b | c) & (b & (c | (a & c)))
/// ```
///
/// Gives terms:
///
/// ```
/// a
/// b | c
/// b
/// c | (a & c)
/// ```
pub struct MintermIter<'a> {
    stack: UnitVec<Node>,
    expr_arena: &'a Arena<AExpr>,
}

impl Iterator for MintermIter<'_> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        let mut top = self.stack.pop()?;

        while let AExpr::BinaryExpr {
            left,
            op: Operator::And | Operator::LogicalAnd,
            right,
        } = self.expr_arena.get(top)
        {
            self.stack.push(*right);
            top = *left;
        }

        Some(top)
    }
}

impl<'a> MintermIter<'a> {
    pub fn new(root: Node, expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            stack: unitvec![root],
            expr_arena,
        }
    }
}
