use polars_core::frame::column::Column;
use polars_core::prelude::{AnyValue, Field};
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_plan::dsl::Expr;
use polars_plan::prelude::apply_multiple;

use crate::compiler::CompiledProgram;

pub(crate) fn into_expr(
    inputs: Box<[Expr]>,
    execute: impl Fn(&mut [Column]) -> PolarsResult<Column> + Send + Sync + 'static,
) -> Expr {
    apply_multiple(
        execute,
        inputs.into_vec(),
        |_schema, fields| {
            let initial_state = &fields[0];
            Ok(Field::new(
                initial_state.name().clone(),
                initial_state.dtype.clone().materialize_unknown(false)?,
            ))
        },
        true,
    )
}

pub(crate) fn execute(
    compiled: CompiledProgram,
    initial_state: &Column,
    row_inputs: &[Column],
) -> PolarsResult<Column> {
    polars_ensure!(initial_state.len() == 1 && row_inputs.len() == 1, InvalidOperation:
        "foldv received invalid inputs");
    let initial_state = match initial_state.get(0)? {
        AnyValue::Int64(value) => value,
        value => polars_bail!(InvalidOperation:
            "foldv expected Int64 state, got {}", value.dtype()),
    };
    let values = row_inputs[0].i64()?.rechunk();
    let values = values.cont_slice()?;
    let pointers = [values.as_ptr()];
    let mut output = vec![0_i64; values.len()];
    // SAFETY: The input, state, output, and generated function ABI all use Int64.
    unsafe {
        (compiled.function)(
            pointers.as_ptr(),
            values.len(),
            &initial_state,
            output.as_mut_ptr(),
        )
    };
    Ok(Column::new(row_inputs[0].name().clone(), output))
}
