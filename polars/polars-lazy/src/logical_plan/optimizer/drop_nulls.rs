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
                // We want to make sure we find this pattern
                // A != null AND B != null AND C != null .. etc.
                // the outer expression always is a binary and operation and the inner
                let iter = (&*expr_arena).iter(*predicate);
                let is_binary_and = |e: &AExpr| {
                    matches!(
                        e,
                        &AExpr::BinaryExpr {
                            op: Operator::And,
                            ..
                        }
                    )
                };
                let is_not_null = |e: &AExpr| matches!(e, &AExpr::IsNotNull(_));
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::stack_opt::OptimizationRule;
    use crate::tests::fruits_cars;
    use crate::utils::test::optimize_lp;

    #[test]
    fn test_drop_nulls_optimization() -> Result<()> {
        let mut rules: Vec<Box<dyn OptimizationRule>> = vec![Box::new(ReplaceDropNulls {})];
        let df = fruits_cars();

        for subset in [
            Some(vec![col("fruits")]),
            Some(vec![col("fruits"), col("cars")]),
            Some(vec![col("fruits"), col("cars"), col("A")]),
            None,
        ] {
            let lp = df.clone().lazy().drop_nulls(subset).logical_plan;
            let out = optimize_lp(lp, &mut rules);
            assert!(matches!(out, LogicalPlan::Udf { .. }));
        }
        Ok(())
    }

    #[test]
    fn test_filter() -> Result<()> {
        // This tests if the filter does not accidentally is optimized by ReplaceNulls

        let data = vec![
            None,
            None,
            None,
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            None,
        ];
        let series = Series::new("data", data);
        let df = DataFrame::new(vec![series])?;

        let column_name = "data";
        let shift_col_1 = col(column_name)
            .shift_and_fill(1, lit(true))
            .lt(col(column_name));
        let shift_col_neg_1 = col(column_name).shift(-1).lt(col(column_name));

        let out = df
            .lazy()
            .with_columns(vec![
                shift_col_1.alias("shift_1"),
                shift_col_neg_1.alias("shift_neg_1"),
            ])
            .with_column(col("shift_1").and(col("shift_neg_1")).alias("diff"))
            .filter(col("diff"))
            .collect()?;
        assert_eq!(out.shape(), (5, 4));

        Ok(())
    }
}
