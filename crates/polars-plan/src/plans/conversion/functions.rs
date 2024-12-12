use arrow::legacy::error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;

use super::*;
use crate::dsl::{Expr, FunctionExpr};
use crate::plans::AExpr;
use crate::prelude::FunctionOptions;

pub(super) fn convert_functions(
    input: Vec<Expr>,
    function: FunctionExpr,
    options: FunctionOptions,
    arena: &mut Arena<AExpr>,
    state: &mut ConversionContext,
) -> PolarsResult<Node> {
    match function {
        // This can be created by col(*).is_null() on empty dataframes.
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal) if input.is_empty() => {
            return to_aexpr_impl(lit(true), arena, state);
        },
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal) if input.is_empty() => {
            return to_aexpr_impl(lit(false), arena, state);
        },
        // Convert to binary expression as the optimizer understands those.
        // Don't exceed 128 expressions as we might stackoverflow.
        FunctionExpr::Boolean(BooleanFunction::AllHorizontal) => {
            if input.len() < 128 {
                let single = input.len() == 1;
                let mut expr = input.into_iter().reduce(|l, r| l.logical_and(r)).unwrap();
                if single {
                    expr = expr.cast(DataType::Boolean)
                }
                return to_aexpr_impl(expr, arena, state);
            }
        },
        FunctionExpr::Boolean(BooleanFunction::AnyHorizontal) => {
            if input.len() < 128 {
                let single = input.len() == 1;
                let mut expr = input.into_iter().reduce(|l, r| l.logical_or(r)).unwrap();
                if single {
                    expr = expr.cast(DataType::Boolean)
                }
                return to_aexpr_impl(expr, arena, state);
            }
        },
        _ => {},
    }

    let e = to_expr_irs(input, arena)?;

    if state.output_name.is_none() {
        // Handles special case functions like `struct.field`.
        if let Some(name) = function.output_name() {
            state.output_name = name
        } else {
            set_function_output_name(&e, state, || format_pl_smallstr!("{}", &function));
        }
    }

    let ae_function = AExpr::Function {
        input: e,
        function,
        options,
    };
    Ok(arena.add(ae_function))
}
