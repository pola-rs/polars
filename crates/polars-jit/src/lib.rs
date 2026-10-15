//! Experimental shape of the Polars JIT crate.

mod compiler;
mod jit_program;
mod planning;
mod runtime;
mod translate;

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

/// Carry state across all rows referenced by `function` and return each updated state.
pub fn foldv(initial_state: Expr, function: Expr) -> PolarsResult<Expr> {
    // Input planning: extract the function body and its row inputs.
    let plan = planning::plan(function)?;
    let planning::PlannedFunction {
        function,
        row_names,
        row_inputs,
    } = plan;

    let mut inputs = Vec::with_capacity(row_inputs.len() + 1);
    inputs.push(initial_state);
    inputs.extend(row_inputs);

    // Runtime integration: Polars evaluates those inputs, then calls this function.
    Ok(runtime::into_expr(
        inputs.into_boxed_slice(),
        move |columns| {
            let (initial_state, row_inputs) = columns.split_first().ok_or_else(|| {
            polars_error::polars_err!(InvalidOperation: "foldv did not receive its initial state")
        })?;

            // Translation: Polars expression + concrete dtypes -> typed JIT program.
            let translated =
                translate::translate(&function, &row_names, initial_state, row_inputs)?;

            // Compilation: typed JIT program + loop bindings -> native list operation.
            let compiled = compiler::compile(
                &translated.program,
                compiler::CompilationShape::Lists(translated.bindings),
            )?;

            // Execution: run the native operation over the supplied columns.
            runtime::execute(compiled, initial_state, row_inputs)
        },
    ))
}
