use super::*;

#[test]
#[cfg(feature = "unique_counts")]
fn test_list_broadcast() {
    // simply test if this runs
    df![
        "g" => [1, 1, 1],
        "a" => [1, 2, 3],
    ]
    .unwrap()
    .lazy()
    .groupby([col("g")])
    .agg([col("a").unique_counts() * count()])
    .collect()
    .unwrap();
}

#[test]
fn ternary_expand_sizes() -> Result<()> {
    let df = df! {
        "a" => [Some("a1"), None, None],
        "b" => [Some("b1"), Some("b2"), None]
    }?;
    let out = df
        .lazy()
        .with_column(
            when(not(lit(true)))
                .then(lit("unexpected"))
                .when(not(col("a").is_null()))
                .then(col("a"))
                .when(not(col("b").is_null()))
                .then(col("b"))
                .otherwise(lit("otherwise"))
                .alias("c"),
        )
        .collect()?;
    let vals = out
        .column("c")?
        .utf8()?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(vals, &["a1", "b2", "otherwise"]);
    Ok(())
}
