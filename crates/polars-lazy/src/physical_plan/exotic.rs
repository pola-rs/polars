use polars_core::prelude::*;
use polars_expr::{create_physical_expr, ExpressionConversionState};

use crate::prelude::*;

#[cfg(feature = "pivot")]
pub(crate) fn prepare_eval_expr(expr: Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::Column(_) => Expr::Column(Arc::from("")),
        Expr::Nth(_) => Expr::Column(Arc::from("")),
        e => e,
    })
}

pub(crate) fn prepare_expression_for_context(
    name: &str,
    expr: &Expr,
    dtype: &DataType,
    ctxt: Context,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    let mut lp_arena = Arena::with_capacity(8);
    let mut expr_arena = Arena::with_capacity(10);

    // create a dummy lazyframe and run a very simple optimization run so that
    // type coercion and simplify expression optimizations run.
    let column = Series::full_null(name, 0, dtype);
    let lf = column
        .into_frame()
        .lazy()
        .without_optimizations()
        .with_simplify_expr(true)
        .select([expr.clone()]);
    let optimized = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(optimized);
    let aexpr = lp.get_exprs().pop().unwrap();

    create_physical_expr(
        &aexpr,
        ctxt,
        &expr_arena,
        None,
        &mut ExpressionConversionState::new(true, 0),
    )
}
