//! Translation of a Polars fold body into a typed `JitProgram`.
//!
//! `resolve` establishes the state and row-input types, and the `Translator` then turns the
//! body into JIT nodes.

mod helpers;
mod resolve;
mod translator;
mod types;

use polars_core::frame::column::Column;
use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;
use translator::Translator;
pub(crate) use types::TranslationResult;

use crate::compiler::ListBindings;
use crate::jit_program::InputId;

pub(crate) fn translate(
    function: &Expr,
    row_names: &[PlSmallStr],
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<TranslationResult> {
    // Input binding: validate the columns supplied by Polars.
    resolve::validate_inputs(row_names, initial_state, row_inputs)?;

    // Type resolution: establish the concrete types used by the program.
    let resolved = resolve::resolve_types(function, row_names, initial_state, row_inputs)?;

    // Program construction: translate the expression into typed JIT nodes.
    let program = Translator::new(&resolved.schema, &row_names[0])
        .expression(&resolved.expression, "root")?;
    Ok(TranslationResult {
        program,
        bindings: ListBindings {
            state_input: InputId(0),
            row_inputs: vec![InputId(1)].into_boxed_slice(),
        },
    })
}
