use super::*;

/// Create a column of length `n` containing `n` copies of the literal `value`.
///
/// Generally you won't need this function, as `lit(value)` already represents a column containing
/// only `value` whose length is automatically set to the correct number of rows.
pub fn repeat<E: Into<Expr>>(value: E, n: Expr) -> Expr {
    let function = |s: Series, n: Series| {
        polars_ensure!(
            n.dtype().is_integer(),
            SchemaMismatch: "expected expression of dtype 'integer', got '{}'", n.dtype()
        );
        let first_value = n.get(0)?;
        let n = first_value.extract::<usize>().ok_or_else(
            || polars_err!(ComputeError: "could not parse value '{}' as a size.", first_value),
        )?;
        Ok(Some(s.new_from_index(0, n)))
    };
    apply_binary(value.into(), n, function, GetOutput::same_type()).alias("repeat")
}
