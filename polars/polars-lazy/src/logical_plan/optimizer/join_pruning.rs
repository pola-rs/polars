use crate::prelude::stack_opt::OptimizationRule;
use crate::prelude::*;
use crate::utils::{equal_aexprs, remove_duplicate_aexprs};
use std::fs::canonicalize;

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
            }
            // Only do this optimization of join keys are equal
            if equal_aexprs(left_on, right_on, expr_arena) => {
                combine_lp_nodes(*input_left, *input_right, lp_arena, expr_arena)
            }
            _ => None
        }
    }
}

/// Recursively combine nodes in the LogicalPlan based on the conditions
/// listed above.
fn combine_lp_nodes(
    input_l: Node,
    input_r: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<ALogicalPlan> {
    use ALogicalPlan::*;

    match (lp_arena.get(input_l), lp_arena.get(input_r)) {
        (
            Aggregate {
                input: child_input_l,
                keys: keys_l,
                aggs: aggs_l,
                apply: apply_l,
                ..
            },
            Aggregate {
                input: child_input_r,
                keys: keys_r,
                aggs: aggs_r,
                apply: apply_r,
                ..
            },
            // skip if we have custom functions
        ) if {
            apply_l.is_none()
                && apply_r.is_none()
                // check if aggregation keys can be combined.
                && equal_aexprs(keys_l, keys_r, expr_arena)
        }
        =>
            {
                let keys = keys_l.clone();
                let aggs = aggs_l
                    .iter()
                    .copied()
                    .chain(aggs_r.iter().copied())
                    .collect();

                combine_lp_nodes(*child_input_l, *child_input_r, lp_arena, expr_arena)
                    .map(|input| {
                        let node = lp_arena.add(input);
                        ALogicalPlanBuilder::new(node, expr_arena, lp_arena)
                            .groupby(keys, aggs, None)
                            .build()

                    })

            }
        (Projection {input: child_input_l, expr: expr_l, ..},
            Projection {input: child_input_r, expr: expr_r, ..})
        // check if all previous nodes/ transformations are equal
        if ALogicalPlan::eq(*child_input_l, *child_input_r, lp_arena, expr_arena)
        => {
            let exprs: Vec<_> = expr_l.iter().copied().chain(expr_r.iter().copied()).collect();
            let exprs = remove_duplicate_aexprs(&exprs, expr_arena);

            combine_lp_nodes(*child_input_l, *child_input_r, lp_arena, expr_arena)
                .map(|input| {
                    let node = lp_arena.add(input);
                    ALogicalPlanBuilder::new(node, expr_arena, lp_arena)
                        .project(exprs)
                        .build()

                })
        }
        #[cfg(feature = "csv-file")]
        (CsvScan {path: path_l,
            schema,
            options: options_l,
            predicate,
            aggregate,
        },
            CsvScan {path: path_r, options: options_r, ..})
        if canonicalize(path_l).unwrap() == canonicalize(path_r).unwrap()
        => {
            let mut options_l = options_l.clone();
            let path = path_l.clone();
            let with_columns = match (&options_l.with_columns, &options_r.with_columns) {
                (Some(l), Some(r)) => Some(l.iter().cloned().chain(r.iter().cloned()).collect()),
                (Some(l), None) => Some(l.clone()),
                (None, Some(r)) => Some(r.clone()),
                _ => None
            };
            options_l.with_columns = with_columns;

            Some(CsvScan {
                path,
                schema: schema.clone(),
                options: options_l,
                predicate: *predicate,
                aggregate: aggregate.clone(),
            })


        }
        _ => {
            if ALogicalPlan::eq(input_l, input_r, lp_arena, expr_arena) {
                Some(lp_arena.take(input_l))
            } else {
                None
            }
        },
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
            "b" => [1, 1, 2, 2, 2],
            "c" => [1, 1, 2, 2, 2]
        )?;

        // // Only aggregation
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

        let (root, mut expr_arena, mut alp_arena) = q1.left_join(q2, col("b"), col("b")).into_alp();
        dbg!(alp_arena.get(root));
        let mut opt = JoinPrune {};
        let out = opt
            .optimize_plan(&mut alp_arena, &mut expr_arena, root)
            .unwrap();
        assert!(matches!(out, ALogicalPlan::Aggregate { .. }));

        // Projection and Aggregations, this needs recursion.
        let q1 = df
            .clone()
            .lazy()
            .select(vec![col("a"), col("b")])
            .groupby(vec![col("b")])
            .agg(vec![col("a").first()]);

        let q2 = df
            .clone()
            .lazy()
            .select(vec![col("a"), col("b"), col("c")])
            .groupby(vec![col("b")])
            .agg(vec![col("a").last()]);

        let (root, mut expr_arena, mut alp_arena) = q1.left_join(q2, col("b"), col("b")).into_alp();
        dbg!(alp_arena.get(root));
        let mut opt = JoinPrune {};
        let out = opt
            .optimize_plan(&mut alp_arena, &mut expr_arena, root)
            .unwrap();
        assert!(matches!(out, ALogicalPlan::Aggregate { .. }));
        Ok(())
    }
}
