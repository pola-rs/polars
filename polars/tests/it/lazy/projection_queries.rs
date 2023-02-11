use polars::prelude::*;

#[test]
fn test_sum_after_filter() -> PolarsResult<()> {
    let df = df![
        "ids" => 0..10,
        "values" => 10..20,
    ]?
    .lazy()
    .filter(not(col("ids").eq(lit(5))))
    .select([col("values").sum()])
    .collect()?;

    assert_eq!(df.column("values")?.get(0)?, AnyValue::Int32(130));
    Ok(())
}

#[test]
fn test_swap_rename() -> PolarsResult<()> {
    let df = df![
        "a" => [1],
        "b" => [2],
    ]?
    .lazy()
    .rename(["a", "b"], ["b", "a"])
    .collect()?;

    let expected = df![
        "b" => [1],
        "a" => [2],
    ]?;
    assert!(df.frame_equal(&expected));
    Ok(())
}

#[test]
fn test_outer_join_with_column_2988() -> PolarsResult<()> {
    let ldf1 = df![
        "key1" => ["foo", "bar"],
        "key2" => ["foo", "bar"],
        "val1" => [3, 1]
    ]?
    .lazy();

    let ldf2 = df![
        "key1" => ["bar", "baz"],
        "key2" => ["bar", "baz"],
        "val2" => [6, 8]
    ]?
    .lazy();

    let out = ldf1
        .join(
            ldf2,
            [col("key1"), col("key2")],
            [col("key1"), col("key2")],
            JoinType::Outer,
        )
        .with_columns([col("key1")])
        .collect()?;
    assert_eq!(out.get_column_names(), &["key1", "key2", "val1", "val2"]);
    assert_eq!(
        Vec::from(out.column("key1")?.utf8()?),
        &[Some("bar"), Some("baz"), Some("foo")]
    );
    assert_eq!(
        Vec::from(out.column("key2")?.utf8()?),
        &[Some("bar"), Some("baz"), Some("foo")]
    );
    assert_eq!(
        Vec::from(out.column("val1")?.i32()?),
        &[Some(1), None, Some(3)]
    );
    assert_eq!(
        Vec::from(out.column("val2")?.i32()?),
        &[Some(6), Some(8), None]
    );

    Ok(())
}

#[test]
fn test_err_no_found() {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [None, Some("a"), Some("b")]
    ]
    .unwrap();

    assert!(matches!(
        df.lazy().filter(col("nope").gt(lit(2))).collect(),
        Err(PolarsError::ColumnNotFound(_))
    ));
}

#[test]
fn test_many_aliasing_projections_5070() -> PolarsResult<()> {
    let df = df! {
        "date" => [1, 2, 3],
        "val" => [1, 2, 3],
    }?;

    let out = df
        .lazy()
        .filter(col("date").gt(lit(1)))
        .select([col("*")])
        .with_columns([col("val").max().alias("max")])
        .with_column(col("max").alias("diff"))
        .with_column((col("val") / col("diff")).alias("output"))
        .select([all().exclude(&["max", "diff"])])
        .collect()?;
    let expected = df![
        "date" => [2, 3],
        "val" => [2, 3],
        "output" => [0, 1],
    ]?;
    assert!(out.frame_equal(&expected));

    Ok(())
}

#[test]
fn test_projection_5086() -> PolarsResult<()> {
    let df = df![
        "a" => ["a", "a", "a", "b"],
        "b" => [1, 0, 1, 0],
        "c" => [0, 1, 2, 0],
    ]?;

    let out = df
        .lazy()
        .select([
            col("a"),
            col("b").take("c").cumsum(false).over([col("a")]).gt(lit(0)),
        ])
        .select([
            col("a"),
            col("b")
                .xor(col("b").shift(1).over([col("a")]))
                .fill_null(lit(true))
                .alias("keep"),
        ])
        .collect()?;

    let expected = df![
        "a" => ["a", "a", "a", "b"],
        "keep" => [true, false, false, true]
    ]?;

    assert!(out.frame_equal(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-struct")]
fn test_unnest_pushdown() -> PolarsResult<()> {
    let df = df![
        "collection" => Series::full_null("", 1, &DataType::Int32),
        "users" => Series::full_null("", 1, &DataType::List(Box::new(DataType::Struct(vec![Field::new("email", DataType::Utf8)])))),
    ]?;

    let out = df
        .lazy()
        .explode(["users"])
        .unnest(["users"])
        .select([col("email")])
        .collect()?;

    assert_eq!(out.get_column_names(), &["email"]);

    Ok(())
}
