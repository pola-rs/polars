//! Input validation and type resolution with Polars' type rules.

use polars_core::frame::column::Column;
use polars_core::prelude::Schema;
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

use crate::jit_program::ScalarType;

/// The column name that stands for the carried state while the body is typed and translated.
pub(super) const STATE_NAME: &str = "__POLARS_JIT_STATE";

/// The concrete types a body is translated with.
pub(super) struct ResolvedTypes {
    /// The body, with the state placeholder replaced by the `STATE_NAME` column.
    pub(super) expression: Expr,
    pub(super) state_type: ScalarType,
    pub(super) row_type: ScalarType,
}

pub(super) fn validate_inputs(
    row_names: &[PlSmallStr],
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<()> {
    polars_ensure!(initial_state.len() == 1, InvalidOperation:
        "foldv initial state must contain exactly one value");
    polars_ensure!(row_names.len() == 1 && row_inputs.len() == 1,
        InvalidOperation: "foldv currently requires exactly one row input");
    polars_ensure!(initial_state.null_count() == 0 && row_inputs[0].null_count() == 0,
        InvalidOperation: "foldv currently requires non-null values");
    Ok(())
}

pub(super) fn resolve_types(
    function: &Expr,
    row_names: &[PlSmallStr],
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<ResolvedTypes> {
    let state_type = ScalarType::from_dtype(initial_state.dtype())?;
    let row_type = ScalarType::from_dtype(row_inputs[0].dtype())?;
    let expression = function.clone().map_expr(|expr| match expr {
        Expr::Element => Expr::Column(STATE_NAME.into()),
        expr => expr,
    });
    let mut schema = Schema::with_capacity(2);
    schema.with_column(STATE_NAME.into(), state_type.dtype());
    schema.with_column(row_names[0].clone(), row_type.dtype());
    let output = expression.to_field(&schema)?.dtype;
    polars_ensure!(output == state_type.dtype(), InvalidOperation:
        "foldv function resolves to {output}, but the state type is {} and row type is {}: {expression:?}",
        state_type.dtype(), row_type.dtype());
    Ok(ResolvedTypes {
        expression,
        state_type,
        row_type,
    })
}
