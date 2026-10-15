use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

use super::PlannedFunction;

pub(super) fn inspect(function: Expr) -> PolarsResult<PlannedFunction> {
    // Expr::Element is the loop-carried state placeholder. Ordinary columns are
    // row inputs and are evaluated by Polars before execution.
    let mut row_names = Vec::<PlSmallStr>::new();
    let _ = function.clone().map_expr(|expr| {
        if let Expr::Column(name) = &expr
            && !row_names.contains(name)
        {
            row_names.push(name.clone());
        }
        expr
    });

    let row_inputs = row_names
        .iter()
        .cloned()
        .map(Expr::Column)
        .collect::<Box<[_]>>();

    Ok(PlannedFunction {
        function,
        row_names: row_names.into_boxed_slice(),
        row_inputs,
    })
}
