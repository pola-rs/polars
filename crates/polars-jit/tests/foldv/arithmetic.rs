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
fn foldv_ignores_aliases_when_translating_the_calculation() -> PolarsResult<()> {
    // Aliases do not change the scalar calculation represented by the program.
    let function = Expr::Element.alias("state") + col("l").alias("value");

    let result = df!("l" => [1_i64, 2])?
        .lazy()
        .select([foldv(lit(10_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(11), Some(13)]
    );
    Ok(())
}

#[test]
fn foldv_subtracts_each_row_from_the_carried_state() -> PolarsResult<()> {
    // Subtraction preserves operand order during every state update.
    let result = df!("l" => [1_i64, 2])?
        .lazy()
        .select([
            foldv(lit(10_i64).cast(DataType::Int64), Expr::Element - col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(result.column("result")?.i64()?.to_vec(), [Some(9), Some(7)]);
    Ok(())
}

#[test]
fn foldv_multiplies_the_carried_state_by_each_row() -> PolarsResult<()> {
    // Multiplication feeds its product into the following iteration.
    let result = df!("l" => [2_i64, 3])?
        .lazy()
        .select([
            foldv(lit(4_i64).cast(DataType::Int64), Expr::Element * col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(8), Some(24)]
    );
    Ok(())
}
