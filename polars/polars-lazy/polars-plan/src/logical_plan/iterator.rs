use crate::prelude::*;
macro_rules! push_expr {
    ($current_expr:expr, $push:ident, $iter:ident) => {{
        use Expr::*;
        match $current_expr {
            Nth(_) | Column(_) | Literal(_) | Wildcard | Columns(_) | DtypeColumn(_) | Count => {}
            Alias(e, _) => $push(e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                $push(right);
                $push(left);
            }
            Cast { expr, .. } => $push(expr),
            Sort { expr, .. } => $push(expr),
            Take { expr, idx } => {
                $push(idx);
                $push(expr);
            }
            Filter { input, by } => {
                $push(by);
                // latest, so that it is popped first
                $push(input);
            }
            SortBy { expr, by, .. } => {
                for e in by {
                    $push(e)
                }
                // latest, so that it is popped first
                $push(expr);
            }
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
                    List(e) => $push(e),
                    Count(e) => $push(e),
                    Quantile { expr, .. } => $push(expr),
                    Sum(e) => $push(e),
                    AggGroups(e) => $push(e),
                    Std(e, _) => $push(e),
                    Var(e, _) => $push(e),
                }
            }
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                $push(predicate);
                $push(falsy);
                // latest, so that it is popped first
                $push(truthy);
            }
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            AnonymousFunction { input, .. } => input.$iter().rev().for_each(|e| $push(e)),
            Function { input, .. } => input.$iter().rev().for_each(|e| $push(e)),
            Explode(e) => $push(e),
            Window {
                function,
                partition_by,
                order_by,
                ..
            } => {
                for e in partition_by.into_iter().rev() {
                    $push(e)
                }
                if let Some(e) = order_by {
                    $push(e);
                }
                // latest so that it is popped first
                $push(function);
            }
            Slice {
                input,
                offset,
                length,
            } => {
                $push(length);
                $push(offset);
                // latest, so that it is popped first
                $push(input);
            }
            Exclude(e, _) => $push(e),
            KeepName(e) => $push(e),
            RenameAlias { expr, .. } => $push(expr),
        }
    }};
}

impl Expr {
    /// Expr::mutate().apply(fn())
    pub fn mutate(&mut self) -> ExprMut {
        let mut stack = Vec::with_capacity(4);
        stack.push(self);
        ExprMut { stack }
    }
}

pub struct ExprMut<'a> {
    stack: Vec<&'a mut Expr>,
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
        while let Some(current_expr) = self.stack.pop() {
            // the order is important, we first modify the Expr
            // before we push its children on the stack.
            // The modification can make the children invalid.
            if !f(current_expr) {
                break;
            }
            let mut push = |e: &'a mut Expr| self.stack.push(e);
            push_expr!(current_expr, push, iter_mut);
        }
    }
}

pub struct ExprIter<'a> {
    stack: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIter<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|current_expr| {
            let mut push = |e: &'a Expr| self.stack.push(e);

            push_expr!(current_expr, push, iter);
            current_expr
        })
    }
}

impl<'a> IntoIterator for &'a Expr {
    type Item = &'a Expr;
    type IntoIter = ExprIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::with_capacity(4);
        stack.push(self);
        ExprIter { stack }
    }
}

impl AExpr {
    /// Push nodes at this level to a pre-allocated stack
    pub(crate) fn nodes<'a>(&'a self, container: &mut Vec<Node>) {
        let mut push = |e: &'a Node| container.push(*e);
        use AExpr::*;

        match self {
            Nth(_) | Column(_) | Literal(_) | Wildcard | Count => {}
            Alias(e, _) => push(e),
            BinaryExpr { left, op: _, right } => {
                // reverse order so that left is popped first
                push(right);
                push(left);
            }
            Cast { expr, .. } => push(expr),
            Sort { expr, .. } => push(expr),
            Take { expr, idx } => {
                push(idx);
                // latest, so that it is popped first
                push(expr);
            }
            SortBy { expr, by, .. } => {
                for node in by {
                    push(node)
                }
                // latest, so that it is popped first
                push(expr);
            }
            Filter { input, by } => {
                push(by);
                // latest, so that it is popped first
                push(input);
            }
            Agg(agg_e) => {
                use AAggExpr::*;
                match agg_e {
                    Max { input, .. } => push(input),
                    Min { input, .. } => push(input),
                    Mean(e) => push(e),
                    Median(e) => push(e),
                    NUnique(e) => push(e),
                    First(e) => push(e),
                    Last(e) => push(e),
                    List(e) => push(e),
                    Count(e) => push(e),
                    Quantile { expr, .. } => push(expr),
                    Sum(e) => push(e),
                    AggGroups(e) => push(e),
                    Std(e, _) => push(e),
                    Var(e, _) => push(e),
                }
            }
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                push(predicate);
                push(falsy);
                // latest, so that it is popped first
                push(truthy);
            }
            AnonymousFunction { input, .. } | Function { input, .. } =>
            // we iterate in reverse order, so that the lhs is popped first and will be found
            // as the root columns/ input columns by `_suffix` and `_keep_name` etc.
            {
                input.iter().rev().for_each(push)
            }
            Explode(e) => push(e),
            Window {
                function,
                partition_by,
                order_by,
                options: _,
            } => {
                for e in partition_by.iter().rev() {
                    push(e);
                }
                if let Some(e) = order_by {
                    push(e);
                }
                // latest so that it is popped first
                push(function);
            }
            Slice {
                input,
                offset,
                length,
            } => {
                push(length);
                push(offset);
                // latest so that it is popped first
                push(input);
            }
        }
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

#[cfg(test)]
mod test {
    use polars_core::df;
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_lp_iter() -> PolarsResult<()> {
        let df = df! {
            "a" => [1, 2]
        }?;

        let (root, lp_arena, _expr_arena) = df
            .lazy()
            .sort("a", Default::default())
            .groupby([col("a")])
            .agg([col("a").first()])
            .logical_plan
            .into_alp();

        let cnt = (&lp_arena).iter(root).count();
        assert_eq!(cnt, 3);
        Ok(())
    }
}
