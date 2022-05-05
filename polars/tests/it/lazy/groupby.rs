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

#[test]
fn test_filter_after_tail() -> Result<()> {
    let df = df![
        "a" => ["foo", "foo", "bar"],
        "b" => [1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("a")])
        .tail(Some(1))
        .filter(col("b").eq(lit(3)))
        .with_predicate_pushdown(false)
        .collect()?;

    let expected = df![
        "a" => ["bar"],
        "b" => [3]
    ]?;
    assert!(out.frame_equal(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "unique_counts")]
fn test_list_arithmetic_in_groupby() -> Result<()> {
    // specifically make the amount of groups equal to df height.
    let df = df![
        "a" => ["foo", "ham", "bar"],
        "b" => [1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("a")])
        .agg([
            col("b").list().alias("original"),
            (col("b").list() * lit(2)).alias("mult_lit"),
            (col("b").list() / lit(2)).alias("div_lit"),
            (col("b").list() - lit(2)).alias("min_lit"),
            (col("b").list() + lit(2)).alias("plus_lit"),
            (col("b").list() % lit(2)).alias("mod_lit"),
            (lit(1) + col("b").list()).alias("lit_plus"),
            (col("b").unique_counts() + count()).alias("plus_count"),
        ])
        .collect()?;

    let cols = ["mult_lit", "div_lit", "plus_count"];
    let out = out.explode(&cols)?.select(&cols)?;

    assert!(out.frame_equal(&df![
        "mult_lit" => [2, 4, 6],
        "div_lit"=> [0, 1, 1],
        "plus_count" => [2 as IdxSize, 2, 2]
    ]?));

    Ok(())
}
