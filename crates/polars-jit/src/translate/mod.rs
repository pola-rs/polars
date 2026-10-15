//! Translation of a Polars fold body into a typed `JitProgram`.
//!
//! `resolve` establishes the state and row-input types using Polars' own type rules, and the
//! `Translator` then turns the body into JIT nodes.

mod helpers;
mod resolve;
mod translator;
mod types;

use polars_core::frame::column::Column;
use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;
pub(crate) use resolve::state_dtype;
use translator::Translator;
pub(crate) use types::TranslationResult;

use crate::compiler::{InputBinding, ListBindings};
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
    let program = Translator::new(row_names, &resolved).expression(
        &resolved.expression,
        resolved.root_type,
        "root",
    )?;

    // The state is input 0; row inputs follow in order.
    let rows = resolved
        .row_types
        .iter()
        .enumerate()
        .map(|(index, dtype)| InputBinding {
            input: InputId(index as u32 + 1),
            dtype: *dtype,
        })
        .collect();
    Ok(TranslationResult {
        program,
        bindings: ListBindings {
            state: InputBinding {
                input: InputId(0),
                dtype: resolved.state_type,
            },
            rows,
        },
    })
}
