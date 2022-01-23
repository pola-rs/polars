use super::*;

#[test]
fn test_pearson_corr() -> Result<()> {
    let df = df! {
        "uid" => [0, 0, 0, 1, 1, 1],
        "day" => [1, 2, 4, 1, 2, 3],
        "cumcases" => [10, 12, 15, 25, 30, 41]
    }
    .unwrap();

    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("uid")])
        // a double aggregation expression.
        .agg([pearson_corr(col("day"), col("cumcases")).alias("pearson_corr")])
        .collect()?;
    let s = out.column("pearson_corr")?.f64()?;
    assert!((s.get(0).unwrap() - 0.997176).abs() < 0.000001);
    assert!((s.get(1).unwrap() - 0.977356).abs() < 0.000001);

    let out = df
        .lazy()
        .groupby_stable([col("uid")])
        // a double aggregation expression.
        .agg([pearson_corr(col("day"), col("cumcases"))
            .pow(2.0)
            .alias("pearson_corr")])
        .collect()
        .unwrap();
    let s = out.column("pearson_corr")?.f64()?;
    assert!((s.get(0).unwrap() - 0.994360902255639).abs() < 0.000001);
    assert!((s.get(1).unwrap() - 0.9552238805970149).abs() < 0.000001);
    Ok(())
}
