//! Input validation and type resolution.

use polars_core::frame::column::Column;
use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

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

pub(super) fn resolve_types(initial_state: &Column, row_inputs: &[Column]) -> PolarsResult<()> {
    polars_ensure!(initial_state.dtype() == &DataType::Int64, InvalidOperation:
        "foldv currently requires an Int64 initial state, got {}", initial_state.dtype());
    polars_ensure!(row_inputs[0].dtype() == &DataType::Int64, InvalidOperation:
        "foldv currently requires an Int64 row input");
    Ok(())
}
