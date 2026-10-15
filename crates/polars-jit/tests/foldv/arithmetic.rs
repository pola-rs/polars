use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_adds_each_row_to_the_carried_state() -> PolarsResult<()> {
    // Each result becomes the state supplied to the following row.
    let result = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([
            foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(3), Some(6)]
    );
    Ok(())
}

#[test]
fn foldv_passes_each_updated_state_to_the_next_row() -> PolarsResult<()> {
    // Starting at zero, each value becomes the next decimal digit.
    let function = Expr::Element * lit(10_i64) + col("l");

    let result = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(12), Some(123)]
    );
    Ok(())
}

#[test]
fn foldv_translates_aliases_subtraction_and_multiplication() -> PolarsResult<()> {
    // Aliases do not change the scalar calculation represented by the program.
    let function = (Expr::Element.alias("state") - col("l").alias("value")) * lit(2_i64);

    let result = df!("l" => [1_i64, 2])?
        .lazy()
        .select([foldv(lit(10_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(18), Some(32)]
    );
    Ok(())
}
