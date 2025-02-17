use super::*;

#[test]
fn test_fold_wildcard() -> PolarsResult<()> {
    let df1 = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df1
        .clone()
        .lazy()
        .select([fold_exprs(lit(0), |a, b| (&a + &b).map(Some), [col("*")]).alias("foo")])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("foo")?.i32()?),
        &[Some(2), Some(4), Some(6)]
    );

    // test if we don't panic due to wildcard
    let _out = df1
        .lazy()
        .select([polars_lazy::dsl::all_horizontal([col("*").is_not_null()])?])
        .collect()?;
    Ok(())
}
