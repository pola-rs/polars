use polars_core::prelude::*;
use polars_expr::{ExpressionConversionState, create_physical_expr};

use crate::prelude::*;

#[cfg(feature = "pivot")]
pub(crate) fn contains_column_refs(expr: &Expr) -> bool {
    for e in expr.into_iter() {
        match e {
            Expr::Column(c) if !c.eq(&PlSmallStr::EMPTY) => return true,
            Expr::Nth(_) => return true,
            Expr::Columns(_) => return true,
            Expr::DtypeColumn(_) => return true,
            Expr::IndexColumn(_) => return true,
            Expr::Selector(_) => return true,
            Expr::Exclude(_, _) => return true,
            Expr::Wildcard => return true,
            Expr::Field(_) => return true,
            _ => {},
        }
    }
    false
}

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
