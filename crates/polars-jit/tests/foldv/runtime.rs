use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_returns_an_empty_column_when_the_row_input_is_empty() -> PolarsResult<()> {
    // With no rows, there are no updated states to return.
    let result = df!("l" => Vec::<i64>::new())?
        .lazy()
        .select([
            foldv(lit(42_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert!(result.column("result")?.is_empty());
    Ok(())
}
