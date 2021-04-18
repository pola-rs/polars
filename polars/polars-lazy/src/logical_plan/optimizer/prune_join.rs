use crate::logical_plan::iterator::*;
use crate::prelude::stack_opt::OptimizationRule;
use crate::prelude::*;
use crate::utils::equal_aexprs;
use std::fs::canonicalize;

fn same_src(a: Node, b: Node, lp_arena: &Arena<ALogicalPlan>) -> bool {
    let (root_a, _) = (&*lp_arena).iter(a).last().unwrap();
    let (root_b, _) = (&*lp_arena).iter(b).last().unwrap();
    let lp_a = lp_arena.get(root_a);
    let lp_b = lp_arena.get(root_b);

    use ALogicalPlan::*;
    match (lp_a, lp_b) {
        (CsvScan { path: path_a, .. }, CsvScan { path: path_b, .. }) => {
            canonicalize(path_a).unwrap() == canonicalize(path_b).unwrap()
        }
        #[cfg(feature = "parquet")]
        (ParquetScan { path: path_a, .. }, ParquetScan { path: path_b, .. }) => {
            canonicalize(path_a).unwrap() == canonicalize(path_b).unwrap()
        }
        (
            DataFrameScan {
                df: df_a,
                schema: schema_a,
                ..
            },
            DataFrameScan {
                df: df_b,
                schema: schema_b,
                ..
            },
        ) => {
            // This can still fail. Maybe add a ptr check?
            df_a.height() == df_b.height()
                && schema_a.eq(schema_b)
                && df_a.get_row(0) == df_b.get_row(0)
        }
        _ => false,
    }
}

// See: https://github.com/ritchie46/polars/issues/449
pub struct PruneJoin {}

impl OptimizationRule for PruneJoin {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        use ALogicalPlan::*;
        match lp {
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } if same_src(*input_left, *input_right, lp_arena)
                && equal_aexprs(left_on, right_on, expr_arena) =>
            {
                match (lp_arena.get(*input_left), lp_arena.get(*input_right)) {
                    (
                        Aggregate {
                            input,
                            keys: keys_left,
                            aggs: aggs_left,
                            apply: apply_left,
                            ..
                        },
                        Aggregate {
                            keys: keys_right,
                            aggs: aggs_right,
                            apply: apply_right,
                            ..
                        },
                    ) if apply_left.is_none()
                        && apply_right.is_none()
                        && equal_aexprs(keys_left, keys_right, expr_arena) =>
                    {
                        let keys = keys_left.clone();
                        let aggs = aggs_left
                            .iter()
                            .copied()
                            .chain(aggs_right.iter().copied())
                            .collect();
                        Some(
                            ALogicalPlanBuilder::new(*input, expr_arena, lp_arena)
                                .groupby(keys, aggs, None)
                                .build(),
                        )
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
