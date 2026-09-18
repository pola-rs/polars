// used only if feature="dtype-duration", "dtype-struct"
#[allow(unused_imports)]
use polars_core::SINGLE_LOCK;
#[cfg(feature = "rank")]
use polars_core::series::ops::NullBehavior;

use super::*;

#[test]
#[cfg(feature = "rank")]
fn test_filter_sort_diff_2984() -> PolarsResult<()> {
    // make sure that sort does not oob if filter returns no values
    let df = df![
    "group"=> ["A" ,"A", "A", "B", "B", "B", "B"],
    "id"=> [1, 2, 1, 4, 5, 4, 6],
    ]?;

    let out = df
        .lazy()
        // don't use stable in this test, it hides wrong state
        .group_by([col("group")])
        .agg([col("id")
            .filter(col("id").lt(lit(3)))
            .sort(Default::default())
            .diff(lit(1), Default::default())
            .sum()])
        .sort(["group"], Default::default())
        .collect()?;

    assert_eq!(Vec::from(out.column("id")?.i32()?), &[Some(1), Some(0)]);
    Ok(())
}

#[test]
fn test_filter_after_tail() -> PolarsResult<()> {
    let df = df![
        "a" => ["foo", "foo", "bar"],
        "b" => [1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .group_by_stable([col("a")])
        .tail(Some(1))
        .filter(col("b").eq(lit(3)))
        .with_predicate_pushdown(false)
        .collect()?;

    let expected = df![
        "a" => ["bar"],
        "b" => [3]
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "diff")]
fn test_filter_diff_arithmetic() -> PolarsResult<()> {
    let df = df![
        "user" => [1, 1, 1, 1, 2],
        "group" => [1, 2, 1, 1, 2],
        "value" => [1, 5, 14, 17, 20]
    ]?;

    let out = df
        .lazy()
        .group_by([col("user")])
        .agg([(col("value")
            .filter(col("group").eq(lit(1)))
            .diff(lit(1), Default::default())
            * lit(2))
        .alias("diff")])
        .sort(["user"], Default::default())
        .explode(
            cols(["diff"]),
            ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            },
        )
        .collect()?;

    let out = out.column("diff")?;
    assert_eq!(
        out,
        &Column::new("diff".into(), &[None, Some(26), Some(6), None])
    );

    Ok(())
}

#[test]
fn test_group_by_lit_agg() -> PolarsResult<()> {
    let df = df![
        "group" => [1, 2, 1, 1, 2],
    ]?;

    let out = df
        .lazy()
        .group_by([col("group")])
        .agg([lit("foo").alias("foo")])
        .collect()?;

    assert_eq!(out.column("foo")?.dtype(), &DataType::String);

    Ok(())
}

#[test]
#[cfg(feature = "diff")]
fn test_group_by_agg_list_with_not_aggregated() -> PolarsResult<()> {
    let df = df![
    "group" => ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
    "value" => [0, 2, 3, 6, 2, 4, 7, 9, 3, 4, 6, 7, ],
    ]?;

    let out = df
        .lazy()
        .group_by([col("group")])
        .agg([
            when(col("value").diff(lit(1), NullBehavior::Ignore).gt_eq(0))
                .then(col("value").diff(lit(1), NullBehavior::Ignore))
                .otherwise(col("value")),
        ])
        .sort(["group"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    let out = out.explode(ExplodeOptions {
        empty_as_null: true,
        keep_nulls: true,
    })?;
    assert_eq!(
        out,
        Column::new("value".into(), &[0, 2, 1, 3, 2, 2, 7, 2, 3, 1, 2, 1])
    );
    Ok(())
}

#[test]
#[cfg(feature = "dtype-decimal")]
fn test_logical_mean_partitioned_group_by_block() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock();
    let df = df![
        "decimal" => [1, 1, 2],
    ]?;

    let out = df
        .lazy()
        .with_column(col("decimal").cast(DataType::Decimal(38, 2)))
        .group_by([col("decimal")])
        .agg([col("decimal").mean().alias("decimal_mean")])
        .sort(["decimal"], Default::default())
        .collect()?;

    let decimal = out.column("decimal")?;

    assert_eq!(decimal.get(0)?, AnyValue::Decimal(100, 38, 2));

    Ok(())
}

#[test]
fn test_filter_aggregated_expression() -> PolarsResult<()> {
    let df: DataFrame = df![
    "day" => [2, 2, 2, 2, 2, 2, 1, 1],
    "y" => [Some(4), Some(5), Some(8), Some(7), Some(9), None, None, None],
    "x" => [1, 2, 3, 4, 5, 6, 1, 2],
    ]?;

    let f = col("y").is_not_null().and(col("x").is_not_null());

    let df = df
        .lazy()
        .group_by([col("day")])
        .agg([(col("x") - col("x").first()).filter(f)])
        .sort(["day"], Default::default())
        .collect()
        .unwrap();
    let x = df.column("x")?;

    assert_eq!(
        x.get(1).unwrap(),
        AnyValue::List(Series::new("".into(), [0, 1, 2, 3, 4]))
    );
    Ok(())
}

#[test]
fn test_group_by() -> PolarsResult<()> {
    let df = df![
        "date" => ["2020-08-21", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-22"],
        "temp" => [20, 10, 7, 9, 1],
        "rain" => [0.2, 0.1, 0.3, 0.1, 0.01],
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("date")])
        .agg([col("temp").count().alias("temp_count")])
        .collect()?;
    assert_eq!(
        out.column("temp_count")?,
        &Column::new("temp_count".into(), [2 as IdxSize, 2, 1])
    );

    // Aggregate multiple columns.
    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("date")])
        .agg([
            col("temp").mean().alias("temp_mean"),
            col("rain").mean().alias("rain_mean"),
        ])
        .collect()?;
    assert_eq!(
        out.column("temp_mean")?,
        &Column::new("temp_mean".into(), [15.0f64, 4.0, 9.0])
    );

    // Group by multiple keys.
    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("date"), col("temp")])
        .agg([col("rain").mean().alias("rain_mean")])
        .collect()?;
    assert!(out.column("rain_mean").is_ok());

    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("date")])
        .agg([col("temp").sum().alias("temp_sum")])
        .collect()?;
    assert_eq!(
        out.column("temp_sum")?,
        &Column::new("temp_sum".into(), [30, 8, 9])
    );

    // Implicitly aggregate all non-key columns.
    let out = df
        .lazy()
        .group_by_stable([col("date")])
        .agg([all().as_expr().n_unique()])
        .collect()?;
    assert_eq!(out.width(), 3);

    Ok(())
}

#[test]
fn test_static_group_by_by_12_columns() {
    // Build GroupBy DataFrame.
    let s0 = Column::new("G1".into(), ["A", "A", "B", "B", "C"].as_ref());
    let s1 = Column::new("N".into(), [1, 2, 2, 4, 2].as_ref());
    let s2 = Column::new("G2".into(), ["k", "l", "m", "m", "l"].as_ref());
    let s3 = Column::new("G3".into(), ["a", "b", "c", "c", "d"].as_ref());
    let s4 = Column::new("G4".into(), ["1", "2", "3", "3", "4"].as_ref());
    let s5 = Column::new("G5".into(), ["X", "Y", "Z", "Z", "W"].as_ref());
    let s6 = Column::new("G6".into(), [false, true, true, true, false].as_ref());
    let s7 = Column::new("G7".into(), ["r", "x", "q", "q", "o"].as_ref());
    let s8 = Column::new("G8".into(), ["R", "X", "Q", "Q", "O"].as_ref());
    let s9 = Column::new("G9".into(), [1, 2, 3, 3, 4].as_ref());
    let s10 = Column::new("G10".into(), [".", "!", "?", "?", "/"].as_ref());
    let s11 = Column::new("G11".into(), ["(", ")", "@", "@", "$"].as_ref());
    let s12 = Column::new("G12".into(), ["-", "_", ";", ";", ","].as_ref());

    let df =
        DataFrame::new_infer_height(vec![s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12])
            .unwrap();

    let keys = [
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12",
    ];
    let out = df
        .lazy()
        .group_by(keys.map(col))
        .agg([col("N").sum()])
        .collect()
        .unwrap();

    assert_eq!(
        Vec::from(&out.column("N").unwrap().i32().unwrap().sort(false)),
        &[Some(1), Some(2), Some(2), Some(6)]
    );
}

#[test]
fn test_dynamic_group_by_by_13_columns() {
    // The content for every group_by series.
    let series_content = ["A", "A", "B", "B", "C"];

    // The name of every group_by series.
    let series_names = [
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13",
    ];

    // Vector to contain every series.
    let mut columns = Vec::with_capacity(14);

    // Create a series for every group name.
    for series_name in series_names {
        let group_columns = Column::new(series_name.into(), series_content.as_ref());
        columns.push(group_columns);
    }

    // Create a series for the aggregation column.
    let agg_series = Column::new("N".into(), [1, 2, 3, 3, 4].as_ref());
    columns.push(agg_series);

    // Create the dataframe with the computed series.
    let df = DataFrame::new_infer_height(columns).unwrap();

    // Aggregate by the 13 columns defined in `series_names`.
    let out = df
        .lazy()
        .group_by(series_names.map(col))
        .agg([col("N").sum()])
        .collect()
        .unwrap();

    // Check that the results of the group-by are correct. The content of every column
    // is equal, then, the grouped columns shall be equal and in the same order.
    for series_name in &series_names {
        assert_eq!(
            Vec::from(&out.column(series_name).unwrap().str().unwrap().sort(false)),
            &[Some("A"), Some("B"), Some("C")]
        );
    }

    // Check the aggregated column is the expected one.
    assert_eq!(
        Vec::from(&out.column("N").unwrap().i32().unwrap().sort(false)),
        &[Some(3), Some(4), Some(6)]
    );
}

#[test]
fn test_group_by_floats() {
    let df = df! {"flt" => [1., 1., 2., 2., 3.],
                "val" => [1, 1, 1, 1, 1]
    }
    .unwrap();
    let out = df
        .lazy()
        .group_by([col("flt")])
        .agg([col("val").sum()])
        .sort(["flt"], SortMultipleOptions::default())
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("val").unwrap().i32().unwrap()),
        &[Some(2), Some(2), Some(1)]
    );
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_group_by_categorical() {
    let mut df = df! {"foo" => ["a", "a", "b", "b", "c"],
                "ham" => ["a", "a", "b", "b", "c"],
                "bar" => [1, 1, 1, 1, 1]
    }
    .unwrap();

    df.apply("foo", |s| {
        s.cast(&DataType::from_categories(Categories::global()))
            .unwrap()
    })
    .unwrap();

    // check multiple keys and categorical
    let out = df
        .lazy()
        .group_by_stable([col("foo"), col("ham")])
        .agg([col("bar").sum()])
        .collect()
        .unwrap();

    assert_eq!(
        Vec::from(
            out.column("bar")
                .unwrap()
                .as_materialized_series()
                .i32()
                .unwrap()
        ),
        &[Some(2), Some(2), Some(1)]
    );
}

#[test]
fn test_group_by_null_handling() -> PolarsResult<()> {
    let df = df!(
        "a" => ["a", "a", "a", "b", "b"],
        "b" => [Some(1), Some(2), None, None, Some(1)]
    )?;
    let out = df
        .lazy()
        .group_by_stable([col("a")])
        .agg([col("b").mean()])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("b")?.as_materialized_series().f64()?),
        &[Some(1.5), Some(1.0)]
    );
    Ok(())
}

#[test]
fn test_group_by_var() -> PolarsResult<()> {
    // check variance and proper coercion to f64
    let df = df![
        "g" => ["foo", "foo", "bar"],
        "flt" => [1.0, 2.0, 3.0],
        "int" => [1, 2, 3]
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("g")])
        .agg([col("int").var(1)])
        .collect()?;
    assert_eq!(out.column("int")?.f64()?.get(0), Some(0.5));

    let out = df
        .lazy()
        .group_by_stable([col("g")])
        .agg([col("int").std(1)])
        .collect()?;
    let val = out.column("int")?.f64()?.get(0).unwrap();
    assert!((val - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.000001);
    Ok(())
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_group_by_null_group() -> PolarsResult<()> {
    // check if null is own group
    let mut df = df![
        "g" => [Some("foo"), Some("foo"), Some("bar"), None, None],
        "flt" => [1.0, 2.0, 3.0, 1.0, 1.0],
        "int" => [1, 2, 3, 1, 1]
    ]?;

    df.try_apply("g", |s| {
        s.cast(&DataType::from_categories(Categories::global()))
    })?;

    let df = df
        .lazy()
        .group_by([col("g")])
        .agg([col("flt").sum(), col("int").sum()])
        .collect()?;
    let expected = df![
        "g"=> [Option::<&str>::None],
        "flt" => [2.0],
        "int" => [2]]
    .unwrap();
    assert_eq!(
        df.lazy().filter(col("g").is_null()).collect().unwrap(),
        expected
    );
    Ok(())
}
