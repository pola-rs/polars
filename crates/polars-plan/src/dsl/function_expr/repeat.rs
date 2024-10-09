use polars_core::prelude::{polars_ensure, polars_err, Column, PolarsResult};

pub fn repeat(args: &[Column]) -> PolarsResult<Column> {
    let c = &args[0];
    let n = &args[1];

    polars_ensure!(
        n.dtype().is_integer(),
        SchemaMismatch: "expected expression of dtype 'integer', got '{}'", n.dtype()
    );

    let first_value = n.get(0)?;
    let n = first_value.extract::<usize>().ok_or_else(
        || polars_err!(ComputeError: "could not parse value '{}' as a size.", first_value),
    )?;

    Ok(c.new_from_index(0, n))
}
