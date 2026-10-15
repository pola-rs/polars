//! Experimental shape of the Polars JIT crate.

mod compiler;
mod jit_program;
mod planning;
mod runtime;
mod translate;

use polars_core::prelude::DataType;
use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

/// Carry state across all rows referenced by `function` and return each updated state.
///
/// The result is named after the initial state, like `pl.fold`.
pub fn foldv(initial_state: Expr, function: Expr) -> PolarsResult<Expr> {
    fold(None, initial_state, function)
}

/// Carry state across the rows of `input` and return each updated state.
///
/// `function` reads the current value of `input` through [`foldv_value`]. Polars evaluates
/// `input` before the fold, so it may be any expression, and its length sets the number of rows
/// even when `function` ignores it. The result is named after `input`, like other expression
/// methods.
pub fn foldv_over(input: Expr, initial_state: Expr, function: Expr) -> PolarsResult<Expr> {
    fold(Some(input), initial_state, function)
}

/// The current value of the input in a [`foldv_over`] function.
pub fn foldv_value() -> Expr {
    Expr::Column(planning::VALUE_NAME.into())
}

fn fold(input: Option<Expr>, initial_state: Expr, function: Expr) -> PolarsResult<Expr> {
    let input_first = input.is_some();

    // Input planning: extract the function body and its row inputs.
    let plan = planning::plan(function, input)?;
    let planning::PlannedFunction {
        function,
        row_names,
        row_inputs,
    } = plan;

    // Output typing: Polars asks for the output dtype before any input is evaluated.
    let schema_function = function.clone();
    let schema_row_names = row_names.clone();
    let state_dtype = move |initial_state: &DataType, row_types: &[DataType]| {
        translate::state_dtype(
            &schema_function,
            &schema_row_names,
            initial_state,
            row_types,
        )
    };

    // Polars names a function expression after its first input. A method-style fold is named
    // after its input, so that input comes first and is swapped behind the state before use.
    let mut inputs = Vec::with_capacity(row_inputs.len() + 1);
    inputs.push(initial_state);
    inputs.extend(row_inputs);
    if input_first {
        inputs.swap(0, 1);
    }

    // Runtime integration: Polars evaluates those inputs, then calls this function.
    Ok(runtime::into_expr(
        inputs.into_boxed_slice(),
        move |columns| {
            if input_first {
                columns.swap(0, 1);
            }
            let (initial_state, row_inputs) = columns.split_first().ok_or_else(|| {
            polars_error::polars_err!(InvalidOperation: "foldv did not receive its initial state")
        })?;

            // Translation: Polars expression + concrete dtypes -> typed JIT program.
            let translated =
                translate::translate(&function, &row_names, initial_state, row_inputs)?;

            // Compilation: typed JIT program + loop bindings -> native list operation.
            let state_type = translated.bindings.state.dtype;
            let compiled = compiler::compile(
                &translated.program,
                compiler::CompilationShape::Lists(translated.bindings),
            )?;

            // Execution: run the native operation over the supplied columns.
            runtime::execute(compiled, state_type, initial_state, row_inputs)
        },
        state_dtype,
        input_first,
    ))
}
