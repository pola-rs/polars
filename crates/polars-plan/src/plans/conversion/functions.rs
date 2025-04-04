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
    mut options: FunctionOptions,
    arena: &mut Arena<AExpr>,
    ctx: &mut ConversionContext,
) -> PolarsResult<Node> {
    use FunctionExpr as F;

    // Return before converting inputs
    match function {
        // This can be created by col(*).is_null() on empty dataframes.
        F::Boolean(BooleanFunction::AllHorizontal) if input.is_empty() => {
            return to_aexpr_impl(lit(true), arena, ctx);
        },
        F::Boolean(BooleanFunction::AnyHorizontal) if input.is_empty() => {
            return to_aexpr_impl(lit(false), arena, ctx);
        },
        // Convert to binary expression as the optimizer understands those.
        // Don't exceed 128 expressions as we might stackoverflow.
        F::Boolean(BooleanFunction::AllHorizontal) => {
            if input.len() < 128 {
                let single = input.len() == 1;
                let mut expr = input.into_iter().reduce(|l, r| l.logical_and(r)).unwrap();
                if single {
                    expr = expr.cast(DataType::Boolean)
                }
                return to_aexpr_impl(expr, arena, ctx);
            }
        },
        F::Boolean(BooleanFunction::AnyHorizontal) => {
            if input.len() < 128 {
                let single = input.len() == 1;
                let mut expr = input.into_iter().reduce(|l, r| l.logical_or(r)).unwrap();
                if single {
                    expr = expr.cast(DataType::Boolean)
                }
                return to_aexpr_impl(expr, arena, ctx);
            }
        },
        _ => {},
    }

    // Converts inputs
    let e = to_expr_irs(input, arena)?;

    match function {
        #[cfg(feature = "diff")]
        F::Diff(_) => {
            polars_ensure!(&e[1].is_scalar(arena), ComputeError: "'n' must be scalar value");
        },
        F::Repeat => {
            polars_ensure!(&e[0].is_scalar(arena), ComputeError: "'value' must be scalar value");
            polars_ensure!(&e[1].is_scalar(arena), ComputeError: "'n' must be scalar value");
        },
        #[cfg(feature = "replace")]
        F::Replace | F::ReplaceStrict { .. } => {
            let old = &e[1];
            let new = &e[1];

            // if old is scalar and new is scalar -> elementwise
            if old.is_scalar(arena) && new.is_scalar(arena) {
                options.set_elementwise();
            }
        },
        F::ShiftAndFill => {
            polars_ensure!(&e[1].is_scalar(arena), ComputeError: "'n' must be scalar value");
            polars_ensure!(&e[2].is_scalar(arena), ComputeError: "'fill_value' must be scalar value");
        },
        _ => {},
    }

    // Validate inputs.
    if function == FunctionExpr::ShiftAndFill {
        polars_ensure!(&e[1].is_scalar(arena), ComputeError: "'n' must be scalar value");
        polars_ensure!(&e[2].is_scalar(arena), ComputeError: "'fill_value' must be scalar value");
    }

    if ctx.output_name.is_none() {
        // Handles special case functions like `struct.field`.
        if let Some(name) = function.output_name() {
            ctx.output_name = name
        } else {
            set_function_output_name(&e, ctx, || format_pl_smallstr!("{}", &function));
        }
    }

    let ae_function = AExpr::Function {
        input: e,
        function,
        options,
    };
    Ok(arena.add(ae_function))
}
