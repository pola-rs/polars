use polars::prelude::*;

#[test]
fn test_sum_after_filter() -> Result<()> {
    let df = df![
        "ids" => 0..10,
        "values" => 10..20,
    ]?
    .lazy()
    .filter(not(col("ids").eq(lit(5))))
    .select([col("values").sum()])
    .collect()?;

    assert_eq!(df.column("values")?.get(0), AnyValue::Int32(130));
    Ok(())
}

#[test]
fn test_swap_rename() -> Result<()> {
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
fn test_outer_join_with_column_2988() -> Result<()> {
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
