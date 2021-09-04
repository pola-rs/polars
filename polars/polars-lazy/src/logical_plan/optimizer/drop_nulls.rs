use crate::logical_plan::iterator::*;
use crate::prelude::stack_opt::OptimizationRule;
use crate::prelude::*;
use crate::utils::aexpr_to_root_names;
use polars_core::prelude::*;
use std::sync::Arc;

/// If we realize that a predicate drops nulls on a subset
/// we replace it with an explicit df.drop_nulls call, as this
/// has a fast path for the no null case
pub struct ReplaceDropNulls {}

impl OptimizationRule for ReplaceDropNulls {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        use ALogicalPlan::*;
        match lp {
            Selection { input, predicate } => {
                if (&*expr_arena).iter(*predicate).all(|(_, e)| {
                    matches!(
                        e,
                        AExpr::IsNotNull(_)
                            | AExpr::BinaryExpr {
                                op: Operator::And,
                                ..
                            }
                            | AExpr::Column(_)
                    )
                }) {
                    let subset = aexpr_to_root_names(*predicate, expr_arena)
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>();

                    let function = move |df: DataFrame| df.drop_nulls(Some(&subset));

                    Some(ALogicalPlan::Udf {
                        input: *input,
                        function: Arc::new(function),
                        // does not matter as this runs after pushdowns have occurred
                        predicate_pd: true,
                        projection_pd: true,
                        schema: None,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
