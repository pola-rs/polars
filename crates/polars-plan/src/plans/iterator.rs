use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;
use visitor::{RewritingVisitor, TreeWalker};

use crate::prelude::*;

macro_rules! push_expr {
    ($current_expr:expr, $c:ident, $push:ident, $push_owned:ident, $iter:ident) => {{
        use Expr::*;
        match $current_expr {
            Nth(_) | Column(_) | Literal(_) | Wildcard | Columns(_) | DtypeColumn(_)
            | IndexColumn(_) | Len => {},
            #[cfg(feature = "dtype-struct")]
            Field(_) => {},
            Alias(e, _) => $push($c, e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                $push($c, right);
                $push($c, left);
            },
            Cast { expr, .. } => $push($c, expr),
            Sort { expr, .. } => $push($c, expr),
            Gather { expr, idx, .. } => {
                $push($c, idx);
                $push($c, expr);
            },
            Filter { input, by } => {
                $push($c, by);
                // latest, so that it is popped first
                $push($c, input);
            },
            SortBy { expr, by, .. } => {
                for e in by {
                    $push_owned($c, e)
                }
                // latest, so that it is popped first
                $push($c, expr);
            },
            Agg(agg_e) => {
                use AggExpr::*;
                match agg_e {
                    Max { input, .. } => $push($c, input),
                    Min { input, .. } => $push($c, input),
                    Mean(e) => $push($c, e),
                    Median(e) => $push($c, e),
                    NUnique(e) => $push($c, e),
                    First(e) => $push($c, e),
                    Last(e) => $push($c, e),
                    Implode(e) => $push($c, e),
                    Count(e, _) => $push($c, e),
                    Quantile { expr, .. } => $push($c, expr),
                    Sum(e) => $push($c, e),
                    AggGroups(e) => $push($c, e),
                    Std(e, _) => $push($c, e),
                    Var(e, _) => $push($c, e),
                }
            },
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                $push($c, predicate);
                $push($c, falsy);
                // latest, so that it is popped first
                $push($c, truthy);
            },
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            AnonymousFunction { input, .. } => input.$iter().rev().for_each(|e| $push_owned($c, e)),
            Function { input, .. } => input.$iter().rev().for_each(|e| $push_owned($c, e)),
            Explode(e) => $push($c, e),
            Window {
                function,
                partition_by,
                ..
            } => {
                for e in partition_by.into_iter().rev() {
                    $push_owned($c, e)
                }
                // latest so that it is popped first
                $push($c, function);
            },
            Slice {
                input,
                offset,
                length,
            } => {
                $push($c, length);
                $push($c, offset);
                // latest, so that it is popped first
                $push($c, input);
            },
            Exclude(e, _) => $push($c, e),
            KeepName(e) => $push($c, e),
            RenameAlias { expr, .. } => $push($c, expr),
            SubPlan { .. } => {},
            // pass
            Selector(_) => {},
        }
    }};
}

pub struct ExprIter<'a> {
    stack: UnitVec<&'a Expr>,
}

impl<'a> Iterator for ExprIter<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack
            .pop()
            .inspect(|current_expr| current_expr.nodes(&mut self.stack))
    }
}

pub struct ExprMapper<F> {
    f: F,
}

impl<F: FnMut(Expr) -> PolarsResult<Expr>> RewritingVisitor for ExprMapper<F> {
    type Node = Expr;
    type Arena = ();

    fn mutate(&mut self, node: Self::Node, _arena: &mut Self::Arena) -> PolarsResult<Self::Node> {
        (self.f)(node)
    }
}

impl Expr {
    pub fn nodes<'a>(&'a self, container: &mut UnitVec<&'a Expr>) {
        let push = |c: &mut UnitVec<&'a Expr>, e: &'a Expr| c.push(e);
        push_expr!(self, container, push, push, iter);
    }

    pub fn nodes_owned(self, container: &mut UnitVec<Expr>) {
        let push_arc = |c: &mut UnitVec<Expr>, e: Arc<Expr>| c.push(Arc::unwrap_or_clone(e));
        let push_owned = |c: &mut UnitVec<Expr>, e: Expr| c.push(e);
        push_expr!(self, container, push_arc, push_owned, into_iter);
    }

    pub fn map_expr<F: FnMut(Self) -> Self>(self, mut f: F) -> Self {
        self.rewrite(&mut ExprMapper { f: |e| Ok(f(e)) }, &mut ())
            .unwrap()
    }

    pub fn try_map_expr<F: FnMut(Self) -> PolarsResult<Self>>(self, f: F) -> PolarsResult<Self> {
        self.rewrite(&mut ExprMapper { f }, &mut ())
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
    stack: UnitVec<Node>,
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
        let stack = unitvec![root];
        AExprIter {
            stack,
            arena: Some(self),
        }
    }
}

pub struct AlpIter<'a> {
    stack: UnitVec<Node>,
    arena: &'a Arena<IR>,
}

pub trait ArenaLpIter<'a> {
    fn iter(&self, root: Node) -> AlpIter<'a>;
}

impl<'a> ArenaLpIter<'a> for &'a Arena<IR> {
    fn iter(&self, root: Node) -> AlpIter<'a> {
        let stack = unitvec![root];
        AlpIter { stack, arena: self }
    }
}

impl<'a> Iterator for AlpIter<'a> {
    type Item = (Node, &'a IR);

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            let lp = self.arena.get(node);
            lp.copy_inputs(&mut self.stack);
            (node, lp)
        })
    }
}
