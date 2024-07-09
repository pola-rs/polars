use polars_core::series::IsSorted;

use super::*;

#[test]
fn test_with_duplicate_column_empty_df() {
    let a = Int32Chunked::from_slice("a", &[]);

    assert_eq!(
        DataFrame::new(vec![a.into_series()])
            .unwrap()
            .lazy()
            .with_columns([lit(true).alias("a")])
            .collect()
            .unwrap()
            .get_column_names(),
        &["a"]
    );
}

#[test]
fn test_drop() -> PolarsResult<()> {
    // dropping all columns is a special case. It may fail because a projection
    // that projects nothing could be misinterpreted as select all.
    let out = df![
        "a" => [1],
    ]?
    .lazy()
    .drop(["a"])
    .collect()?;
    assert_eq!(out.width(), 0);
    Ok(())
}

#[test]
#[cfg(feature = "dynamic_group_by")]
fn test_special_group_by_schemas() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 2, 3, 4, 5],
    ]?;

    let out = df
        .clone()
        .lazy()
        .with_column(col("a").set_sorted_flag(IsSorted::Ascending))
        .rolling(
            col("a"),
            [],
            RollingGroupOptions {
                period: Duration::parse("2i"),
                offset: Duration::parse("0i"),
                closed_window: ClosedWindow::Left,
                ..Default::default()
            },
        )
        .agg([col("b").sum().alias("sum")])
        .select([col("a"), col("sum")])
        .collect()?;

    assert_eq!(
        out.column("sum")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[3, 5, 7, 9, 5]
    );

    let out = df
        .lazy()
        .with_column(col("a").set_sorted_flag(IsSorted::Ascending))
        .group_by_dynamic(
            col("a"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("2i"),
                period: Duration::parse("2i"),
                offset: Duration::parse("0i"),
                label: Label::DataPoint,
                include_boundaries: false,
                closed_window: ClosedWindow::Left,
                ..Default::default()
            },
        )
        .agg([col("b").sum().alias("sum")])
        .select([col("a"), col("sum")])
        .collect()?;

    assert_eq!(
        out.column("sum")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1, 5, 9]
    );

    Ok(())
}

#[test]
fn max_on_empty_df_3027() -> PolarsResult<()> {
    let df = df! {
        "id" => ["1"],
        "name" => ["one"],
        "numb" => [1]
    }?
    .head(Some(0));

    let out = df
        .lazy()
        .group_by(&[col("id"), col("name")])
        .agg(&[col("numb").max()])
        .collect()?;
    assert_eq!(out.shape(), (0, 3));
    Ok(())
}

#[test]
fn test_alias_before_cast() -> PolarsResult<()> {
    let out = df![
        "a" => [1, 2, 3],
    ]?
    .lazy()
    .select([col("a").alias("d").cast(DataType::Int32)])
    .select([all()])
    .collect()?;
    assert_eq!(
        Vec::from(out.column("d")?.i32()?),
        &[Some(1), Some(2), Some(3)]
    );
    Ok(())
}

#[test]
fn test_sorted_path() -> PolarsResult<()> {
    // start with a sorted column and see if the metadata remains preserved

    let payloads = &[1, 2, 3];
    let df = df![
        "a"=> [AnyValue::List(Series::new("", payloads)), AnyValue::List(Series::new("", payloads)), AnyValue::List(Series::new("", payloads))]
    ]?;

    let out = df
        .lazy()
        .with_row_index("index", None)
        .explode(["a"])
        .group_by(["index"])
        .agg([col("a").count().alias("count")])
        .collect()?;

    let s = out.column("index")?;
    assert_eq!(s.is_sorted_flag(), IsSorted::Ascending);

    Ok(())
}

#[test]
fn test_sorted_path_joins() -> PolarsResult<()> {
    let dfa = df![
        "a"=> [1, 2, 3]
    ]?;

    let dfb = df![
        "a"=> [1, 2, 3]
    ]?;

    let out = dfa
        .lazy()
        .with_column(col("a").set_sorted_flag(IsSorted::Ascending))
        .join(dfb.lazy(), [col("a")], [col("a")], JoinType::Left.into())
        .collect()?;

    let s = out.column("a")?;
    assert_eq!(s.is_sorted_flag(), IsSorted::Ascending);

    Ok(())
}

#[test]
fn test_unknown_supertype_ignore() -> PolarsResult<()> {
    let df = df![
        "col1" => [0., 3., 2., 1.],
        "col2" => [0., 0., 1., 1.],
    ]?;

    let out = df
        .lazy()
        .with_columns([(col("col1").fill_null(0f64) + col("col2"))])
        .collect()?;
    assert_eq!(out.shape(), (4, 2));
    Ok(())
}

#[test]
fn test_apply_multiple_columns() -> PolarsResult<()> {
    let df = fruits_cars();

    let multiply = |s: &mut [Series]| (&(&s[0] * &s[0])? * &s[1]).map(Some);

    let out = df
        .clone()
        .lazy()
        .select([map_multiple(
            multiply,
            [col("A"), col("B")],
            GetOutput::from_type(DataType::Float64),
        )])
        .collect()?;
    let out = out.column("A")?;
    let out = out.i32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(5), Some(16), Some(27), Some(32), Some(25)]
    );

    let out = df
        .lazy()
        .group_by_stable([col("cars")])
        .agg([apply_multiple(
            multiply,
            [col("A"), col("B")],
            GetOutput::from_type(DataType::Float64),
            true,
        )])
        .collect()?;

    let out = out.column("A")?;
    let out = out.list()?.get_as_series(1).unwrap();
    let out = out.i32()?;

    assert_eq!(Vec::from(out), &[Some(16)]);
    Ok(())
}

#[test]
fn test_group_by_on_lists() -> PolarsResult<()> {
    let s0 = Series::new("", [1i32, 2, 3]);
    let s1 = Series::new("groups", [4i32, 5]);

    let mut builder =
        ListPrimitiveChunkedBuilder::<Int32Type>::new("arrays", 10, 10, DataType::Int32);
    builder.append_series(&s0).unwrap();
    builder.append_series(&s1).unwrap();
    let s2 = builder.finish().into_series();

    let df = DataFrame::new(vec![s1, s2])?;
    let out = df
        .clone()
        .lazy()
        .group_by([col("groups")])
        .agg([col("arrays").first()])
        .collect()?;

    assert_eq!(
        out.column("arrays")?.dtype(),
        &DataType::List(Box::new(DataType::Int32))
    );

    let out = df
        .lazy()
        .group_by([col("groups")])
        .agg([col("arrays").implode()])
        .collect()?;

    // a list of lists
    assert_eq!(
        out.column("arrays")?.dtype(),
        &DataType::List(Box::new(DataType::List(Box::new(DataType::List(
            Box::new(DataType::Int32)
        )))))
    );

    Ok(())
}
