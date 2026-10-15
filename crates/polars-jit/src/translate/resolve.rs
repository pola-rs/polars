//! Input validation and type resolution.

use polars_core::frame::column::Column;
use polars_core::prelude::{DataType, Schema};
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

/// The column name that stands for the carried state while the body is typed and translated.
pub(super) const STATE_NAME: &str = "__POLARS_JIT_STATE";

/// The body and the schema it is typed against.
pub(super) struct ResolvedTypes {
    /// The body, with the state placeholder replaced by the `STATE_NAME` column.
    pub(super) expression: Expr,
    pub(super) schema: Schema,
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
    polars_ensure!(initial_state.dtype() == &DataType::Int64, InvalidOperation:
        "foldv currently requires an Int64 initial state, got {}", initial_state.dtype());
    polars_ensure!(row_inputs[0].dtype() == &DataType::Int64, InvalidOperation:
        "foldv currently requires an Int64 row input");
    let expression = function.clone().map_expr(|expr| match expr {
        Expr::Element => Expr::Column(STATE_NAME.into()),
        expr => expr,
    });
    let mut schema = Schema::with_capacity(2);
    schema.with_column(STATE_NAME.into(), DataType::Int64);
    schema.with_column(row_names[0].clone(), DataType::Int64);
    polars_ensure!(expression.to_field(&schema)?.dtype == DataType::Int64,
        InvalidOperation: "foldv function must produce Int64");
    Ok(ResolvedTypes { expression, schema })
}
