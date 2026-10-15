//! Type resolution: the fold's state type and the dtype of its body, using Polars' type rules.

use polars_core::frame::column::Column;
use polars_core::prelude::{DataType, Schema};
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

use crate::jit_program::ScalarType;
use crate::translate::helpers::with_arithmetic_types;

/// The column name that stands for the carried state while the body is typed and translated.
pub(super) const STATE_NAME: &str = "__POLARS_JIT_STATE";

/// The concrete types a body is translated with.
pub(super) struct ResolvedTypes {
    /// The body, with the state placeholder replaced by the `STATE_NAME` column.
    pub(super) expression: Expr,
    pub(super) state_type: ScalarType,
    pub(super) row_types: Box<[ScalarType]>,
    /// The dtype the body's untyped nulls take, when the body has no type of its own.
    pub(super) root_type: Option<ScalarType>,
}

pub(super) fn validate_inputs(
    row_names: &[PlSmallStr],
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<()> {
    polars_ensure!(initial_state.len() == 1, InvalidOperation:
        "foldv initial state must contain exactly one value");
    polars_ensure!(!row_inputs.is_empty(), InvalidOperation:
        "foldv function must reference at least one row input");
    polars_ensure!(row_names.len() == row_inputs.len(), InvalidOperation:
        "foldv received inconsistent row-input metadata");
    Ok(())
}

pub(super) fn resolve_types(
    function: &Expr,
    row_names: &[PlSmallStr],
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<ResolvedTypes> {
    let row_types = row_inputs
        .iter()
        .map(|input| ScalarType::from_dtype(input.dtype()))
        .collect::<PolarsResult<Box<[_]>>>()?;
    let row_dtypes = row_types
        .iter()
        .map(|dtype| dtype.dtype())
        .collect::<Vec<_>>();
    let expression = with_state_column(function);
    let state_type = ScalarType::from_dtype(&resolve_state_dtype(
        &expression,
        row_names,
        initial_state.dtype(),
        &row_dtypes,
    )?)?;
    let output = body_dtype(&expression, row_names, &state_type.dtype(), &row_dtypes)?;
    // A body of dtype `Null`, such as `when(...).then(None).otherwise(None)`, is always null;
    // like any untyped null it takes the type it is combined with, here the state type.
    polars_ensure!(output == state_type.dtype() || output.is_null(), InvalidOperation:
        "foldv function resolves to {output}, but the state type is {}: {expression:?}",
        state_type.dtype());
    Ok(ResolvedTypes {
        expression,
        state_type,
        row_types,
        root_type: output.is_null().then_some(state_type),
    })
}

/// The fold's state dtype for a function before Polars has supplied any input columns.
pub(crate) fn state_dtype(
    function: &Expr,
    row_names: &[PlSmallStr],
    initial_state: &DataType,
    row_types: &[DataType],
) -> PolarsResult<DataType> {
    resolve_state_dtype(
        &with_state_column(function),
        row_names,
        initial_state,
        row_types,
    )
}

/// The schema a body is typed against: the state followed by the row inputs.
pub(super) fn body_schema(
    row_names: &[PlSmallStr],
    state: &DataType,
    row_types: &[DataType],
) -> Schema {
    let mut schema = Schema::with_capacity(row_names.len() + 1);
    schema.with_column(STATE_NAME.into(), state.clone());
    for (name, dtype) in row_names.iter().zip(row_types) {
        schema.with_column(name.clone(), dtype.clone());
    }
    schema
}

/// Replace the state placeholder with a named column so the body can be typed against a schema.
fn with_state_column(function: &Expr) -> Expr {
    function.clone().map_expr(|expr| match expr {
        Expr::Element => Expr::Column(STATE_NAME.into()),
        expr => expr,
    })
}

/// The dtype the body produces for the given state and row-input dtypes.
fn body_dtype(
    expression: &Expr,
    row_names: &[PlSmallStr],
    state: &DataType,
    row_types: &[DataType],
) -> PolarsResult<DataType> {
    with_arithmetic_types(expression)
        .to_field(&body_schema(row_names, state, row_types))?
        .dtype
        .materialize_unknown(false)
}

/// The fold's state dtype. Like `pl.lit(None) + pl.col("a")`, an untyped null initial state
/// takes the type the body gives it, because the supertype of `Null` and `T` is `T`.
fn resolve_state_dtype(
    expression: &Expr,
    row_names: &[PlSmallStr],
    initial_state: &DataType,
    row_types: &[DataType],
) -> PolarsResult<DataType> {
    if !initial_state.is_null() {
        return Ok(initial_state.clone());
    }
    let inferred = body_dtype(expression, row_names, &DataType::Null, row_types)?;
    polars_ensure!(!inferred.is_null(), InvalidOperation:
        "foldv initial state is an untyped null and the function does not determine its type; \
        give it a dtype, such as `pl.lit(None, dtype=pl.Int64)`");
    Ok(inferred)
}
