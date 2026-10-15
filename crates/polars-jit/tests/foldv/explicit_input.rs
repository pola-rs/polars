use polars::prelude::*;
use polars_jit::{foldv_over, foldv_value};

#[test]
fn foldv_over_runs_once_per_input_row_when_the_value_is_unused() -> PolarsResult<()> {
    // The input sets the number of rows even if the function never reads its value.
    let result = df!("l" => [10_i64, 20, 30])?
        .lazy()
        .select([foldv_over(
            col("l"),
            lit(0_i64).cast(DataType::Int64),
            Expr::Element + lit(1_i64),
        )?])
        .collect()?;

    assert_eq!(result.get_column_names(), ["l"]);
    assert_eq!(
        result.column("l")?.i64()?.to_vec(),
        [Some(1), Some(2), Some(3)]
    );
    Ok(())
}

#[test]
fn foldv_over_reads_an_input_that_polars_evaluates_first() -> PolarsResult<()> {
    // Sorting is not a JIT operation; Polars evaluates it before the fold reads its values.
    let result = df!("l" => [3_i64, 1, 2])?
        .lazy()
        .select([foldv_over(
            col("l").sort(SortOptions::default()),
            lit(0_i64).cast(DataType::Int64),
            Expr::Element * lit(10_i64) + foldv_value(),
        )?])
        .collect()?;

    assert_eq!(
        result.column("l")?.i64()?.to_vec(),
        [Some(1), Some(12), Some(123)]
    );
    Ok(())
}
