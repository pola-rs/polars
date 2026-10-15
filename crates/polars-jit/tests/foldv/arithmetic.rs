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
