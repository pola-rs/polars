use arrow::legacy::error::PolarsResult;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use crate::prelude::*;

macro_rules! push_expr {
    ($current_expr:expr, $push:ident, $iter:ident) => {{
        use Expr::*;
        match $current_expr {
            Nth(_) | Column(_) | Literal(_) | Wildcard | Columns(_) | DtypeColumn(_) | Len => {},
            Alias(e, _) => $push(e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                $push(right);
                $push(left);
            },
            Cast { expr, .. } => $push(expr),
            Sort { expr, .. } => $push(expr),
            Gather { expr, idx, .. } => {
                $push(idx);
                $push(expr);
            },
            Filter { input, by } => {
                $push(by);
                // latest, so that it is popped first
                $push(input);
            },
            SortBy { expr, by, .. } => {
                for e in by {
                    $push(e)
                }
                // latest, so that it is popped first
                $push(expr);
            },
            Agg(agg_e) => {
                use AggExpr::*;
                match agg_e {
                    Max { input, .. } => $push(input),
                    Min { input, .. } => $push(input),
                    Mean(e) => $push(e),
                    Median(e) => $push(e),
                    NUnique(e) => $push(e),
                    First(e) => $push(e),
                    Last(e) => $push(e),
                    Implode(e) => $push(e),
                    Count(e, _) => $push(e),
                    Quantile { expr, .. } => $push(expr),
                    Sum(e) => $push(e),
                    AggGroups(e) => $push(e),
                    Std(e, _) => $push(e),
                    Var(e, _) => $push(e),
                }
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                $push(predicate);
                $push(falsy);
                // latest, so that it is popped first
                $push(truthy);
            },
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            AnonymousFunction { input, .. } => input.$iter().rev().for_each(|e| $push(e)),
            Function { input, .. } => input.$iter().rev().for_each(|e| $push(e)),
            Explode(e) => $push(e),
            Window {
                function,
                partition_by,
                ..
            } => {
                for e in partition_by.into_iter().rev() {
                    $push(e)
                }
                // latest so that it is popped first
                $push(function);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                $push(length);
                $push(offset);
                // latest, so that it is popped first
                $push(input);
            },
            Exclude(e, _) => $push(e),
            KeepName(e) => $push(e),
            RenameAlias { expr, .. } => $push(expr),
            SubPlan { .. } => {},
            // pass
            Selector(_) => {},
        }
    }};
}

impl Expr {
    /// Expr::mutate().apply(fn())
    pub fn mutate(&mut self) -> ExprMut {
        let stack = unitvec!(self);
        ExprMut { stack }
    }
}

pub struct ExprMut<'a> {
    stack: UnitVec<&'a mut Expr>,
}

impl<'a> ExprMut<'a> {
    ///
    /// # Arguments
    /// * `f` - A function that may mutate an expression. If the function returns `true` iteration
    /// continues.
    pub fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Expr) -> bool,
    {
        let _ = self.try_apply(|e| Ok(f(e)));
    }

    pub fn try_apply<F>(&mut self, mut f: F) -> PolarsResult<()>
    where
        F: FnMut(&mut Expr) -> PolarsResult<bool>,
    {
        while let Some(current_expr) = self.stack.pop() {
            // the order is important, we first modify the Expr
            // before we push its children on the stack.
            // The modification can make the children invalid.
            if !f(current_expr)? {
                break;
            }
            current_expr.nodes_mut(&mut self.stack)
        }
        Ok(())
    }
}

pub struct ExprIter<'a> {
    stack: UnitVec<&'a Expr>,
}

impl<'a> Iterator for ExprIter<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|current_expr| {
            current_expr.nodes(&mut self.stack);
            current_expr
        })
    }
}

impl Expr {
    pub fn nodes<'a>(&'a self, container: &mut UnitVec<&'a Expr>) {
        let mut push = |e: &'a Expr| container.push(e);
        push_expr!(self, push, iter);
    }

    pub fn nodes_mut<'a>(&'a mut self, container: &mut UnitVec<&'a mut Expr>) {
        let mut push = |e: &'a mut Expr| container.push(e);
        push_expr!(self, push, iter_mut);
    }
}

impl<'a> IntoIterator for &'a Expr {
    type Item = &'a Expr;
    type IntoIter = ExprIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let stack = unitvec!(self);
        ExprIter { stack }
    }
}

pub struct AExprIter<'a> {
    stack: Vec<Node>,
    arena: Option<&'a Arena<AExpr>>,
}

impl<'a> Iterator for AExprIter<'a> {
    type Item = (Node, &'a AExpr);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            // take the arena because the bchk doesn't allow a mutable borrow to the field.
            let arena = self.arena.unwrap();
            let current_expr = arena.get(node);
            current_expr.nodes(&mut self.stack);

            self.arena = Some(arena);
            (node, current_expr)
        })
    }
}

pub trait ArenaExprIter<'a> {
    fn iter(&self, root: Node) -> AExprIter<'a>;
}

impl<'a> ArenaExprIter<'a> for &'a Arena<AExpr> {
    fn iter(&self, root: Node) -> AExprIter<'a> {
        let mut stack = Vec::with_capacity(4);
        stack.push(root);
        AExprIter {
            stack,
            arena: Some(self),
        }
    }
}

pub struct AlpIter<'a> {
    stack: Vec<Node>,
    arena: &'a Arena<ALogicalPlan>,
}

pub trait ArenaLpIter<'a> {
    fn iter(&self, root: Node) -> AlpIter<'a>;
}

impl<'a> ArenaLpIter<'a> for &'a Arena<ALogicalPlan> {
    fn iter(&self, root: Node) -> AlpIter<'a> {
        let stack = vec![root];
        AlpIter { stack, arena: self }
    }
}

impl<'a> Iterator for AlpIter<'a> {
    type Item = (Node, &'a ALogicalPlan);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            let lp = self.arena.get(node);
            lp.copy_inputs(&mut self.stack);
            (node, lp)
        })
    }
}
