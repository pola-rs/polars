use super::*;

#[test]
#[cfg(feature = "range")]
fn test_int_range_agg() -> PolarsResult<()> {
    let df = df![
        "x" => [5, 5, 4, 4, 2, 2]
    ]?;

    let out = df
        .lazy()
        .with_columns([int_range(lit(0i32), len(), 1, DataType::Int64).over([col("x")])])
        .collect()?;
    assert_eq!(
        Vec::from_iter(out.column("literal")?.i64()?.into_no_null_iter()),
        &[0, 1, 0, 1, 0, 1]
    );

    Ok(())
}

#[test]
#[cfg(all(feature = "unique_counts", feature = "log"))]
fn test_groups_update() -> PolarsResult<()> {
    let df = df!["group" => ["A" ,"A", "A", "B", "B", "B", "B"],
    "id"=> [1, 1, 2, 3, 4, 3, 5]
    ]?;

    let out = df
        .lazy()
        .group_by_stable([col("group")])
        .agg([col("id").unique_counts().log(2.0)])
        .explode([col("id")])
        .collect()?;
    assert_eq!(
        out.column("id")?
            .f64()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1.0, 0.0, 1.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
#[cfg(feature = "log")]
fn test_groups_update_binary_shift_log() -> PolarsResult<()> {
    let out = df![
        "a" => [1, 2, 3, 5],
        "b" => [1, 2, 1, 2],
    ]?
    .lazy()
    .group_by([col("b")])
    .agg([col("a") - col("a").shift(lit(1)).log(2.0)])
    .sort(["b"], Default::default())
    .explode([col("a")])
    .collect()?;
    assert_eq!(
        Vec::from(out.column("a")?.f64()?),
        &[None, Some(3.0), None, Some(4.0)]
    );

    Ok(())
}

#[test]
#[cfg(feature = "cum_agg")]
fn test_expand_list() -> PolarsResult<()> {
    let out = df![
        "a" => [1, 2],
        "b" => [2, 3],
    ]?
    .lazy()
    .select([cols(["a", "b"]).cum_sum(false)])
    .collect()?;

    let expected = df![
        "a" => [1, 3],
        "b" => [2, 5]
    ]?;

    assert!(out.equals(&expected));

    Ok(())
}

#[test]
fn test_apply_groups_empty() -> PolarsResult<()> {
    let df = df![
        "id" => [1, 1],
        "hi" => ["here", "here"]
    ]?;
    let out = df
        .clone()
        .lazy()
        .filter(col("id").eq(lit(2)))
        .group_by([col("id")])
        .agg([col("hi").drop_nulls().unique()])
        .explain(true)
        .unwrap();
    println!("{}", out);

    let out = df
        .lazy()
        .filter(col("id").eq(lit(2)))
        .group_by([col("id")])
        .agg([col("hi").drop_nulls().unique()])
        .collect()?;

    assert_eq!(
        out.dtypes(),
        &[DataType::Int32, DataType::List(Box::new(DataType::String))]
    );
    assert_eq!(out.shape(), (0, 2));

    Ok(())
}
