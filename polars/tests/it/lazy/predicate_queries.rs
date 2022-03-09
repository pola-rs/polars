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
