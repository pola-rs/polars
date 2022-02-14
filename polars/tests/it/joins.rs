use polars::prelude::*;

#[test]
fn join_nans_outer() -> Result<()> {
    let df1 = df! {
             "w" => [Some(2.5), None, Some(f64::NAN), None, Some(2.5), Some(f64::NAN), None, Some(3.0)],
             "t" => [Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("l")],
            "c" => [Some(10), Some(5), Some(3), Some(2), Some(9), Some(4), Some(11), Some(3)],
        }?
        .lazy();
    let a1 = df1
        .clone()
        .groupby(vec![col("w").alias("w"), col("t").alias("t")])
        .agg(vec![col("c").sum().alias("c_sum")]);
    let a2 = df1
        .groupby(vec![col("w").alias("w"), col("t").alias("t")])
        .agg(vec![col("c").max().alias("c_max")]);

    let res = a1
        .join_builder()
        .with(a2)
        .left_on(vec![col("w").alias("w"), col("t").alias("t")])
        .right_on(vec![col("w").alias("w"), col("t").alias("t")])
        .how(JoinType::Outer)
        .finish()
        .collect()?;

    assert_eq!(res.shape(), (4, 4));
    Ok(())
}
