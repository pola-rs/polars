//! Polars lazy does not implement a pivot because it is impossible to know the schema without
//! materializing the whole dataset. This makes a pivot quite a terrible operation for performant
//! workflows. An optimization can never be pushed down passed a pivot.
//!
//! We can do a pivot on an eager `DataFrame` as that is already materialized. The code for the
//! pivot is here, because we want to be able to pass expressions to the pivot operation.
//!

use polars_core::frame::groupby::expr::PhysicalAggExpr;
use polars_core::prelude::*;
use polars_ops::pivot::PivotAgg;

use crate::physical_plan::exotic::{prepare_eval_expr, prepare_expression_for_context};
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

struct PivotExpr(Expr);

impl PhysicalAggExpr for PivotExpr {
    fn evaluate(&self, df: &DataFrame, groups: &GroupsProxy) -> PolarsResult<Series> {
        let state = ExecutionState::new();
        let dtype = df.get_columns()[0].dtype();
        let phys_expr = prepare_expression_for_context("", &self.0, dtype, Context::Aggregation)?;
        phys_expr
            .evaluate_on_groups(df, groups, &state)
            .map(|mut ac| ac.aggregated())
    }

    fn root_name(&self) -> PolarsResult<&str> {
        Ok("")
    }
}

pub fn pivot<I0, S0, I1, S1, I2, S2>(
    df: &DataFrame,
    values: I0,
    index: I1,
    columns: I2,
    agg_expr: Expr,
    sort_columns: bool,
    // used as separator/delimiter in generated column names.
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    S0: AsRef<str>,
    I1: IntoIterator<Item = S1>,
    S1: AsRef<str>,
    I2: IntoIterator<Item = S2>,
    S2: AsRef<str>,
{
    // make sure that the root column is replaced
    let expr = prepare_eval_expr(agg_expr);
    polars_ops::pivot::pivot(
        df,
        values,
        index,
        columns,
        PivotAgg::Expr(Arc::new(PivotExpr(expr))),
        sort_columns,
        separator,
    )
}

pub fn pivot_stable<I0, S0, I1, S1, I2, S2>(
    df: &DataFrame,
    values: I0,
    index: I1,
    columns: I2,
    agg_expr: Expr,
    sort_columns: bool,
    // used as separator/delimiter in generated column names.
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    S0: AsRef<str>,
    I1: IntoIterator<Item = S1>,
    S1: AsRef<str>,
    I2: IntoIterator<Item = S2>,
    S2: AsRef<str>,
{
    // make sure that the root column is replaced
    let expr = prepare_eval_expr(agg_expr);
    polars_ops::pivot::pivot_stable(
        df,
        values,
        index,
        columns,
        PivotAgg::Expr(Arc::new(PivotExpr(expr))),
        sort_columns,
        separator,
    )
}
