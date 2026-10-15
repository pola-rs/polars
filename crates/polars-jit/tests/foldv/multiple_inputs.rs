use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_reads_multiple_columns_at_the_same_row() -> PolarsResult<()> {
    // Every iteration adds the product of the current values from both columns.
    let result = df!(
        "a" => [1_i64, 2, 3],
        "b" => [4_i64, 5, 6],
    )?
    .lazy()
    .select([foldv(
        lit(0).cast(DataType::Int64),
        Expr::Element + col("a") * col("b"),
    )?
    .alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(4), Some(14), Some(32)]
    );
    Ok(())
}

#[test]
fn foldv_reads_multiple_columns_with_different_dtypes() -> PolarsResult<()> {
    // Each input keeps its own buffer width before Polars coercion converts the calculation.
    let result = df!(
        "a" => [1_i32, 2, 3],
        "b" => [0.5_f64, 1.5, 2.5],
    )?
    .lazy()
    .select([foldv(lit(0.0_f64), Expr::Element + col("a") + col("b"))?.alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.f64()?.to_vec(),
        [Some(1.5), Some(5.0), Some(10.5)]
    );
    Ok(())
}

#[test]
fn foldv_loads_each_referenced_column_only_once_per_row() -> PolarsResult<()> {
    // Reusing a column in the expression reuses its planned input binding.
    let result = df!(
        "a" => [1_i64, 2],
        "b" => [10_i64, 20],
    )?
    .lazy()
    .select([foldv(
        lit(0).cast(DataType::Int64),
        Expr::Element + col("a") + col("b") + col("a"),
    )?
    .alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(12), Some(36)]
    );
    Ok(())
}

#[test]
fn foldv_allows_a_body_that_does_not_read_the_previous_state() -> PolarsResult<()> {
    // Typed bindings declare the state even when no expression node references it.
    let result = df!(
        "a" => [1_i64, 2, 3],
        "b" => [4_i64, 5, 6],
    )?
    .lazy()
    .select([foldv(lit(0).cast(DataType::Int64), col("a") + col("b"))?.alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(5), Some(7), Some(9)]
    );
    Ok(())
}
