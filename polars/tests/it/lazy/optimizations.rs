use super::*;

#[test]
fn test_filter() -> PolarsResult<()> {
    // This tests if the filter does not accidentally is optimized by ReplaceNulls

    let data = vec![
        None,
        None,
        None,
        None,
        Some(false),
        Some(false),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
        Some(false),
        None,
    ];
    let series = Series::new("data", data);
    let df = DataFrame::new(vec![series])?;

    let column_name = "data";
    let shift_col_1 = col(column_name)
        .shift_and_fill(1, lit(true))
        .lt(col(column_name));
    let shift_col_neg_1 = col(column_name).shift(-1).lt(col(column_name));

    let out = df
        .lazy()
        .with_columns(vec![
            shift_col_1.alias("shift_1"),
            shift_col_neg_1.alias("shift_neg_1"),
        ])
        .with_column(col("shift_1").and(col("shift_neg_1")).alias("diff"))
        .filter(col("diff"))
        .collect()?;
    assert_eq!(out.shape(), (5, 4));

    Ok(())
}
