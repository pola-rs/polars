use crate::prelude::*;

pub struct ExprIter<'a> {
    stack: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIter<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|current_expr| {
            use Expr::*;
            let mut push = |e: &'a Expr| self.stack.push(e);

            match current_expr {
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
                Agg(agg_e) => {
                    use AggExpr::*;
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
                Udf { input, .. } => push(input),
                Shift { input, .. } => push(input),
                Reverse(e) => push(e),
                Duplicated(e) => push(e),
                Unique(e) => push(e),
                Explode(e) => push(e),
                Window {
                    function,
                    partition_by,
                    order_by,
                } => {
                    push(function);
                    push(partition_by);
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
                Except(e) => push(e),
            }
            current_expr
        })
    }
}

impl<'a> IntoIterator for &'a Expr {
    type Item = &'a Expr;
    type IntoIter = ExprIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::with_capacity(16);
        stack.push(self);
        ExprIter { stack }
    }
}
