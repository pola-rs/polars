use super::*;

#[test]
fn test_predicate_after_renaming() -> Result<()> {
    let df = df![
        "foo" => [1, 2, 3],
        "bar" => [3, 2, 1]
    ]?
    .lazy()
    .rename(["foo", "bar"], ["foo2", "bar2"])
    .filter(col("foo2").eq(col("bar2")))
    .collect()?;

    let expected = df![
        "foo2" => [2],
        "bar2" => [2],
    ]?;
    assert!(df.frame_equal(&expected));

    Ok(())
}

#[test]
fn filter_true_lit() -> Result<()> {
    let df = df! {
        "a" => [Some(true), Some(false), None],
        "b" => ["1", "2", "3"]
    }?;
    let filter = col("a").eq(lit(true));
    let with_true = df.clone().lazy().filter(filter.clone()).collect()?;
    let with_not_true = df
        .clone()
        .lazy()
        .filter(not(filter.clone()))
        .with_predicate_pushdown(false)
        .with_projection_pushdown(false)
        .collect()?;
    let res = with_true.vstack(&with_not_true)?;
    assert!(res.frame_equal_missing(&df));
    Ok(())
}

#[test]
fn test_combine_columns_in_filter() -> Result<()> {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [None, Some("a"), Some("b")]
    ]?;

    let out = df
        .lazy()
        .filter(cols(vec!["a".to_string(), "b".to_string()]).gt(lit(2)))
        .collect()?;

    let expected = df![
        "a" => [3],
        "b" => ["b"],
    ]?;

    // "b" > "2" == true
    assert!(out.frame_equal(&expected));
    Ok(())
}
