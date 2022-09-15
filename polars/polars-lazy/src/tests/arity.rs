use super::*;

#[test]
fn test_pearson_corr() -> PolarsResult<()> {
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
        .agg([pearson_corr(col("day"), col("cumcases"), 1).alias("pearson_corr")])
        .collect()?;
    let s = out.column("pearson_corr")?.f64()?;
    assert!((s.get(0).unwrap() - 0.997176).abs() < 0.000001);
    assert!((s.get(1).unwrap() - 0.977356).abs() < 0.000001);

    let out = df
        .lazy()
        .groupby_stable([col("uid")])
        // a double aggregation expression.
        .agg([pearson_corr(col("day"), col("cumcases"), 1)
            .pow(2.0)
            .alias("pearson_corr")])
        .collect()
        .unwrap();
    let s = out.column("pearson_corr")?.f64()?;
    assert!((s.get(0).unwrap() - 0.994360902255639).abs() < 0.000001);
    assert!((s.get(1).unwrap() - 0.9552238805970149).abs() < 0.000001);
    Ok(())
}

// TODO! fix this we must get a token that prevents resetting the string cache until the plan has
// finished running. We cannot store a mutexguard in the executionstate because they don't implement
// send.
#[test]
#[cfg(feature = "ignore")]
fn test_single_thread_when_then_otherwise_categorical() -> PolarsResult<()> {
    let df = df!["col1"=> ["a", "b", "a", "b"],
        "col2"=> ["a", "a", "b", "b"],
        "col3"=> ["same", "same", "same", "same"]
    ]?;

    let out = df
        .lazy()
        .with_column(col("*").cast(DataType::Categorical))
        .select([when(col("col1").eq(col("col2")))
            .then(col("col3"))
            .otherwise(col("col1"))])
        .collect()?;
    let col = out.column("col3")?;
    assert_eq!(col.dtype(), &DataType::Categorical);
    let s = format!("{}", col);
    assert!(s.contains("same"));
    Ok(())
}

#[test]
fn test_lazy_ternary() {
    let df = get_df()
        .lazy()
        .with_column(
            when(col("sepal.length").lt(lit(5.0)))
                .then(lit(10))
                .otherwise(lit(1))
                .alias("new"),
        )
        .collect()
        .unwrap();
    assert_eq!(Some(43), df.column("new").unwrap().sum::<i32>());
}
