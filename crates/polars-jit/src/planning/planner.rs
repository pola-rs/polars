use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

use super::PlannedFunction;

/// Column name through which a fold body reads the current value of an explicit input.
pub(crate) const VALUE_NAME: &str = "__POLARS_JIT_VALUE";

pub(super) fn inspect(function: Expr, input: Option<Expr>) -> PolarsResult<PlannedFunction> {
    // Expr::Element is the loop-carried state placeholder. Ordinary columns are
    // row inputs and are evaluated by Polars before execution.
    let mut row_names = Vec::<PlSmallStr>::new();
    let mut row_inputs = Vec::<Expr>::new();

    // An explicit input comes first: Polars evaluates the whole expression, so any expression
    // works as an input, and its length sets the number of rows even if the body ignores it.
    if let Some(input) = input {
        row_names.push(VALUE_NAME.into());
        row_inputs.push(input);
    }
    let _ = function.clone().map_expr(|expr| {
        if let Expr::Column(name) = &expr
            && !row_names.contains(name)
        {
            row_names.push(name.clone());
            row_inputs.push(Expr::Column(name.clone()));
        }
        expr
    });
    let row_inputs = row_inputs.into_boxed_slice();

    Ok(PlannedFunction {
        function,
        row_names: row_names.into_boxed_slice(),
        row_inputs,
    })
}
