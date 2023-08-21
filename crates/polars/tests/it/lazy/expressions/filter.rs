use super::*;

#[test]
fn test_filter_in_group_by_agg() -> PolarsResult<()> {
    // This tests if the filter is correctly handled by the binary expression.
    // This could lead to UB if it were not the case. The filter creates an empty column.
    // but the group tuples could still be untouched leading to out of bounds aggregation.
    let df = df![
        "a" => [1, 1, 2],
        "b" => [1, 2, 3]
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by([col("a")])
        .agg([(col("b").filter(col("b").eq(lit(100))) * lit(2))
            .mean()
            .alias("b_mean")])
        .collect()?;

    assert_eq!(out.column("b_mean")?.null_count(), 2);

    let out = df
        .lazy()
        .group_by([col("a")])
        .agg([(col("b")
            .filter(col("b").eq(lit(100)))
            .map(|v| Ok(Some(v)), GetOutput::same_type()))
        .mean()
        .alias("b_mean")])
        .collect()?;
    assert_eq!(out.column("b_mean")?.null_count(), 2);

    Ok(())
}
