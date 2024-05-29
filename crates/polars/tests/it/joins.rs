#[cfg(feature = "lazy")]
use polars::prelude::*;

#[test]
#[cfg(feature = "lazy")]
fn join_nans_outer() -> PolarsResult<()> {
    let df1 = df! {
             "w" => [Some(2.5), None, Some(f64::NAN), None, Some(2.5), Some(f64::NAN), None, Some(3.0)],
             "t" => [Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("xl"), Some("l")],
            "c" => [Some(10), Some(5), Some(3), Some(2), Some(9), Some(4), Some(11), Some(3)],
        }?
        .lazy();
    let a1 = df1
        .clone()
        .group_by(vec![col("w").alias("w"), col("t")])
        .agg(vec![col("c").sum().alias("c_sum")]);
    let a2 = df1
        .group_by(vec![col("w").alias("w"), col("t")])
        .agg(vec![col("c").max().alias("c_max")]);

    let res = a1
        .join_builder()
        .with(a2)
        .left_on(vec![col("w"), col("t")])
        .right_on(vec![col("w"), col("t")])
        .how(JoinType::Full)
        .coalesce(JoinCoalesce::CoalesceColumns)
        .join_nulls(true)
        .finish()
        .collect()?;

    assert_eq!(res.shape(), (4, 4));
    Ok(())
}

#[test]
#[cfg(feature = "lazy")]
fn join_empty_datasets() -> PolarsResult<()> {
    let a = DataFrame::new(Vec::from([Series::new_empty("foo", &DataType::Int64)])).unwrap();
    let b = DataFrame::new(Vec::from([
        Series::new_empty("foo", &DataType::Int64),
        Series::new_empty("bar", &DataType::Int64),
    ]))
    .unwrap();

    a.lazy()
        .group_by([col("foo")])
        .agg([all().last()])
        .inner_join(b.lazy(), "foo", "foo")
        .collect()
        .unwrap();

    Ok(())
}
