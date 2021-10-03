use crate::prelude::*;

macro_rules! push_expr {
    ($current_expr:expr, $push:ident, $iter:ident) => {{
        use Expr::*;
        match $current_expr {
            Column(_) | Literal(_) | Wildcard | Columns(_) => {}
            Alias(e, _) => $push(e),
            Not(e) => $push(e),
            BinaryExpr { left, op: _, right } => {
                $push(left);
                $push(right);
            }
            IsNull(e) => $push(e),
            IsNotNull(e) => $push(e),
            Cast { expr, .. } => $push(expr),
            Sort { expr, .. } => $push(expr),
            Take { expr, idx } => {
                $push(expr);
                $push(idx);
            }
            Filter { input, by } => {
                $push(input);
                $push(by)
            }
            SortBy { expr, by, .. } => {
                $push(expr);
                $push(by)
            }
            Agg(agg_e) => {
                use AggExpr::*;
                match agg_e {
                    Max(e) => $push(e),
                    Min(e) => $push(e),
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
                    Std(e) => $push(e),
                    Var(e) => $push(e),
                }
            }
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                $push(truthy);
                $push(falsy);
                $push(predicate)
            }
            Function { input, .. } => input.$iter().for_each(|e| $push(e)),
            Shift { input, .. } => $push(input),
            Reverse(e) => $push(e),
            Duplicated(e) => $push(e),
            IsUnique(e) => $push(e),
            Explode(e) => $push(e),
            Window {
                function,
                partition_by,
                order_by,
                ..
            } => {
                $push(function);
                for e in partition_by {
                    $push(e)
                }
                if let Some(e) = order_by {
                    $push(e);
                }
            }
            Slice { input, .. } => $push(input),
            BinaryFunction {
                input_a, input_b, ..
            } => {
                $push(input_a);
                $push(input_b)
            }
            Exclude(e, _) => $push(e),
            KeepName(e) => $push(e),
            SufPreFix { expr, .. } => $push(expr),
        }
    }};
}

impl Expr {
    /// Expr::mutate().apply(fn())
    pub(crate) fn mutate(&mut self) -> ExprMut {
        let mut stack = Vec::with_capacity(8);
        stack.push(self);
        ExprMut { stack }
    }
}

pub(crate) struct ExprMut<'a> {
    stack: Vec<&'a mut Expr>,
}

impl<'a> ExprMut<'a> {
    ///
    /// # Arguments
    /// * `f` - A function that may mutate an expression. If the function returns `true` iteration
    /// continues.
    pub(crate) fn apply<F>(&mut self, f: F)
    where
        F: Fn(&mut Expr) -> bool,
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
        let mut stack = Vec::with_capacity(8);
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
            Column(_) | Literal(_) | Wildcard => {}
            Alias(e, _) => push(e),
            Not(e) => push(e),
            BinaryExpr { left, op: _, right } => {
                push(left);
                push(right);
            }
            IsNull(e) => push(e),
            IsNotNull(e) => push(e),
            Cast { expr, .. } => push(expr),
            Sort { expr, .. } => push(expr),
            Take { expr, idx } => {
                push(expr);
                push(idx);
            }
            SortBy { expr, by, .. } => {
                push(expr);
                push(by);
            }
            Filter { input, by } => {
                push(input);
                push(by);
            }
            Agg(agg_e) => {
                use AAggExpr::*;
                match agg_e {
                    Max(e) => push(e),
                    Min(e) => push(e),
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
                    Std(e) => push(e),
                    Var(e) => push(e),
                }
            }
            Ternary {
                truthy,
                falsy,
                predicate,
            } => {
                push(truthy);
                push(falsy);
                push(predicate)
            }
            Function { input, .. } => input.iter().for_each(push),
            Shift { input, .. } => push(input),
            Reverse(e) => push(e),
            Duplicated(e) => push(e),
            IsUnique(e) => push(e),
            Explode(e) => push(e),
            Window {
                function,
                partition_by,
                order_by,
                options: _,
            } => {
                push(function);
                for e in partition_by {
                    push(e);
                }
                if let Some(e) = order_by {
                    push(e);
                }
            }
            Slice { input, .. } => push(input),
            BinaryFunction {
                input_a, input_b, ..
            } => {
                push(input_a);
                push(input_b)
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

pub(crate) trait ArenaExprIter<'a> {
    fn iter(&self, root: Node) -> AExprIter<'a>;
}

impl<'a> ArenaExprIter<'a> for &'a Arena<AExpr> {
    fn iter(&self, root: Node) -> AExprIter<'a> {
        let mut stack = Vec::with_capacity(8);
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

pub(crate) trait ArenaLpIter<'a> {
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
    use super::*;
    use polars_core::df;
    use polars_core::prelude::*;

    #[test]
    fn test_lp_iter() -> Result<()> {
        let df = df! {
            "a" => [1, 2]
        }?;

        let (root, lp_arena, _expr_arena) = df
            .lazy()
            .sort("a", false)
            .groupby([col("a")])
            .agg([col("a").first()])
            .logical_plan
            .into_alp();

        let cnt = (&lp_arena).iter(root).count();
        assert_eq!(cnt, 3);
        Ok(())
    }
}
