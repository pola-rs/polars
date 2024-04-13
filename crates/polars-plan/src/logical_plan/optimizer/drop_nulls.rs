use super::*;
use crate::logical_plan::iterator::*;

/// If we realize that a predicate drops nulls on a subset
/// we replace it with an explicit df.drop_nulls call, as this
/// has a fast path for the no null case
pub(super) struct ReplaceDropNulls {}

impl OptimizationRule for ReplaceDropNulls {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<IR> {
        let lp = lp_arena.get(node);

        use IR::*;
        match lp {
            Filter { input, predicate } => {
                // We want to make sure we find this pattern
                // A != null AND B != null AND C != null .. etc.
                // the outer expression always is a binary and operation and the inner
                let iter = (&*expr_arena).iter(predicate.node());
                let is_binary_and = |e: &AExpr| {
                    matches!(
                        e,
                        &AExpr::BinaryExpr {
                            op: Operator::And,
                            ..
                        }
                    )
                };
                let is_not_null = |e: &AExpr| {
                    matches!(
                        e,
                        &AExpr::Function {
                            function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                            ..
                        }
                    )
                };
                let is_column = |e: &AExpr| matches!(e, &AExpr::Column(_));
                let is_lit_true =
                    |e: &AExpr| matches!(e, &AExpr::Literal(LiteralValue::Boolean(true)));

                let mut binary_and_count = 0;
                let mut not_null_count = 0;
                let mut column_count = 0;
                for (_, e) in iter {
                    if is_binary_and(e) {
                        binary_and_count += 1;
                    } else if is_column(e) {
                        column_count += 1;
                    } else if is_not_null(e) {
                        not_null_count += 1;
                    } else if is_lit_true(e) {
                        // do nothing
                    } else {
                        // only expected
                        //  - binary and
                        //  - column
                        //  - is not null
                        //  - lit true
                        // so we can return early
                        return None;
                    }
                }
                if not_null_count == column_count && binary_and_count < column_count {
                    let subset = Arc::from(aexpr_to_leaf_names(predicate.node(), expr_arena));

                    Some(IR::MapFunction {
                        input: *input,
                        function: FunctionNode::DropNulls { subset },
                    })
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}
