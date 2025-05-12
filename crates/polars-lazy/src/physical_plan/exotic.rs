use polars_core::prelude::*;
use polars_expr::{ExpressionConversionState, create_physical_expr};

use crate::prelude::*;

// TODO - TBD: drop or keep
// #[cfg(feature = "pivot")]
// pub(crate) fn prepare_eval_expr(expr: Expr) -> Expr {
//     expr.map_expr(|e| match e {
//         Expr::Column(_) => Expr::Column(PlSmallStr::EMPTY),
//         Expr::Nth(_) => Expr::Column(PlSmallStr::EMPTY),
//         e => e,
//     })
// }

pub(crate) fn prepare_expression_for_context(
    name: PlSmallStr,
    expr: &Expr,
    dtype: &DataType,
    ctxt: Context,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let mut lp_arena = Arena::with_capacity(8);
    let mut expr_arena = Arena::with_capacity(10);

    // create a dummy lazyframe and run a very simple optimization run so that
    // type coercion and simplify expression optimizations run.
    let column = Series::full_null(name, 0, dtype);
    let df = column.into_frame();
    let input_schema = df.schema().clone();
    let lf = df
        .lazy()
        .without_optimizations()
        .with_simplify_expr(true)
        .select([expr.clone()]);
    let optimized = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(optimized);
    let aexpr = lp
        .get_exprs()
        .pop()
        .ok_or_else(|| polars_err!(ComputeError: "expected expressions in the context"))?;

    create_physical_expr(
        &aexpr,
        ctxt,
        &expr_arena,
        &input_schema,
        &mut ExpressionConversionState::new(true),
    )
}

pub(crate) fn prepare_expression_for_context_with_schema(
    schema: &SchemaRef,
    expr: &Expr,
    ctxt: Context,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let mut lp_arena = Arena::with_capacity(8); // TBD: what size is needed? or can we safely ignore
    let mut expr_arena = Arena::with_capacity(10); // TBD: what size is needed? or can we safely ignore

    // create a dummy lazyframe and run a very simple optimization run so that
    // type coercion and simplify expression optimizations run.
    let df = DataFrame::full_null(schema, 0);
    let input_schema = df.schema().clone();
    let lf = df
        .lazy()
        .without_optimizations()
        .with_simplify_expr(true)
        .select([expr.clone()]);
    let optimized = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(optimized);
    let aexpr = lp
        .get_exprs()
        .pop()
        .ok_or_else(|| polars_err!(ComputeError: "expected expressions in the context"))?;

    create_physical_expr(
        &aexpr,
        ctxt,
        &expr_arena,
        &input_schema,
        &mut ExpressionConversionState::new(true),
    )
}
