use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_executes_strict_casts_whose_values_fit() -> PolarsResult<()> {
    // A strict cast only fails for valid values that do not fit; nulls stay null.
    let result = df!("l" => [Some(1_i64), None, Some(3)])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int32),
            Expr::Element + col("l").strict_cast(DataType::Int32),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i32()?.to_vec(),
        [Some(1), None, None]
    );
    Ok(())
}

#[test]
fn foldv_reports_the_row_and_value_of_a_failed_strict_cast() -> PolarsResult<()> {
    // The fold stops at the first value that does not fit, like Polars' strict cast error.
    let error = df!("l" => [1_i64, 3_000_000_000, 4_000_000_000])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int32),
            Expr::Element + col("l").strict_cast(DataType::Int32),
        )?])
        .collect()
        .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("conversion from `i64` to `i32` failed in foldv at row 1: [3000000000]"),
        "{message}"
    );
    Ok(())
}

#[test]
fn foldv_reports_nan_in_a_failed_strict_float_cast() -> PolarsResult<()> {
    // NaN has no integer value, so a strict float-to-integer cast fails on it.
    let error = df!("l" => [1.0_f64, f64::NAN])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int64),
            Expr::Element + col("l").strict_cast(DataType::Int64),
        )?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("at row 1: [NaN]"), "{error}");
    Ok(())
}

#[test]
fn foldv_ignores_strict_casts_in_branches_that_are_not_selected() -> PolarsResult<()> {
    // Like Polars, a value that only reaches the unselected branch does not raise.
    let function = when(col("l").lt(lit(10_i64)))
        .then(Expr::Element + col("l").strict_cast(DataType::Int32))
        .otherwise(Expr::Element);
    let result = df!("l" => [1_i64, 3_000_000_000, 2])?
        .lazy()
        .select([foldv(lit(0).cast(DataType::Int32), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i32()?.to_vec(),
        [Some(1), Some(1), Some(3)]
    );
    Ok(())
}
