use super::*;

#[test]
fn test_is_in() -> PolarsResult<()> {
    let df = df![
        "x" => [1, 2, 3],
        "y" => ["a", "b", "c"]
    ]?;
    let s = Series::new("a", ["a", "b"]);

    let out = df
        .lazy()
        .select([col("y").is_in(lit(s)).alias("isin")])
        .collect()?;
    assert_eq!(
        Vec::from(out.column("isin")?.bool()?),
        &[Some(true), Some(true), Some(false)]
    );
    Ok(())
}
