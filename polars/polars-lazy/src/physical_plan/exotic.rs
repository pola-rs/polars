use polars_core::prelude::*;

use crate::physical_plan::planner::create_physical_expr;
use crate::prelude::*;

pub(crate) fn prepare_eval_expr(mut expr: Expr) -> Expr {
    expr.mutate().apply(|e| match e {
        Expr::Column(name) => {
            *name = Arc::from("");
            true
        }
        Expr::Nth(_) => {
            *e = Expr::Column(Arc::from(""));
            true
        }
        _ => true,
    });
    expr
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
    let lf = DataFrame::new_no_checks(vec![column])
        .lazy()
        .without_optimizations()
        .with_simplify_expr(true)
        .select([expr.clone()]);
    let optimized = lf.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    let lp = lp_arena.get(optimized);
    let aexpr = lp.get_exprs().pop().unwrap();

    create_physical_expr(aexpr, ctxt, &expr_arena, None)
}
