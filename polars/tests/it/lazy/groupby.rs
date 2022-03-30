use super::*;

#[test]
fn test_filter_sort_diff_2984() -> Result<()> {
    // make sort that sort doest not oob if filter returns no values
    let df = df![
    "group"=> ["A" ,"A", "A", "B", "B", "B", "B"],
    "id"=> [1, 2, 1, 4, 5, 4, 6],
    ]?;

    let out = df
        .lazy()
        // don't use stable in this test, it hides wrong state
        .groupby([col("group")])
        .agg([col("id")
            .filter(col("id").lt(lit(3)))
            .sort(false)
            .diff(1, Default::default())
            .sum()])
        .sort("group", Default::default())
        .collect()?;

    assert_eq!(Vec::from(out.column("id")?.i32()?), &[Some(1), None]);
    Ok(())
}
