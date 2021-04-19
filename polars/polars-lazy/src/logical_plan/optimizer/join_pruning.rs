use crate::prelude::stack_opt::OptimizationRule;
use crate::prelude::*;
use crate::utils::equal_aexprs;

/// Optimization rule that prunes a join, if the latest operation could be merged and the rest of
/// the LP is equal.
// See: https://github.com/ritchie46/polars/issues/449
pub struct JoinPrune {}

impl OptimizationRule for JoinPrune {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        // We check if:
        // 1: join the same tables,
        // 2: join on the same columns
        // 3: inputs of joins can be combined
        //      * AGGREGATION if keys are equal
        //      * PROJECTION can always be combined.
        // 4: the nodes in the LP before (3) are equal.
        use ALogicalPlan::*;
        match lp {
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } if equal_aexprs(left_on, right_on, expr_arena) => {
                match (lp_arena.get(*input_left), lp_arena.get(*input_right)) {
                    (
                        Aggregate {
                            input: input_l,
                            keys: keys_l,
                            aggs: aggs_l,
                            apply: apply_l,
                            ..
                        },
                        Aggregate {
                            input: input_r,
                            keys: keys_r,
                            aggs: aggs_r,
                            apply: apply_r,
                            ..
                        },
                    // skip if we have custom functions
                    ) if apply_l.is_none()
                        && apply_r.is_none()
                        // check if aggregation keys can be combined.
                        && equal_aexprs(keys_l, keys_r, expr_arena)
                        // check if all previous nodes/ transformations are equal
                        && ALogicalPlan::eq(*input_l, *input_r, lp_arena)
                    =>
                    {
                        let keys = keys_l.clone();
                        let aggs = aggs_l
                            .iter()
                            .copied()
                            .chain(aggs_r.iter().copied())
                            .collect();
                        Some(
                            ALogicalPlanBuilder::new(*input_l, expr_arena, lp_arena)
                                .groupby(keys, aggs, None)
                                .build(),
                        )
                    }
                    (Projection {input: input_l, expr: expr_l, ..},
                        Projection {input: input_r, expr: expr_r, ..})
                    // check if all previous nodes/ transformations are equal
                    if ALogicalPlan::eq(*input_l, *input_r, lp_arena)
                    => {
                        let exprs = expr_l.iter().copied().chain(expr_r.iter().copied()).collect();
                        Some(ALogicalPlanBuilder::new(*input_l, expr_arena, lp_arena)
                            .project(exprs)
                            .build())
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use polars_core::df;
    use polars_core::prelude::*;

    #[test]
    fn test_join_prune() -> Result<()> {
        let df = df!(
            "a" => [1, 2, 3, 4, 5],
            "b" => [1, 1, 2, 2, 2]
        )?;

        let q1 = df
            .clone()
            .lazy()
            .groupby(vec![col("b")])
            .agg(vec![col("a").first()]);

        let q2 = df
            .clone()
            .lazy()
            .groupby(vec![col("b")])
            .agg(vec![col("a").last()]);

        let (root, mut expr_arena, mut alp_arena) =
            q1.left_join(q2, col("b"), col("b"), None).into_alp();
        dbg!(alp_arena.get(root));
        let mut opt = JoinPrune {};
        let out = opt
            .optimize_plan(&mut alp_arena, &mut expr_arena, root)
            .unwrap();
        assert!(matches!(out, ALogicalPlan::Aggregate { .. }));
        Ok(())
    }
}
