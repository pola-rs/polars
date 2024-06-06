#[cfg(feature = "rank")]
use polars_core::series::ops::NullBehavior;
// used only if feature="dtype-duration", "dtype-struct"
#[allow(unused_imports)]
use polars_core::SINGLE_LOCK;

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
            .diff(1, Default::default())
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
            .diff(1, Default::default())
            * lit(2))
        .alias("diff")])
        .sort(["user"], Default::default())
        .explode([col("diff")])
        .collect()?;

    let out = out.column("diff")?;
    assert_eq!(out, &Series::new("diff", &[None, Some(26), Some(6), None]));

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
        .agg([when(col("value").diff(1, NullBehavior::Ignore).gt_eq(0))
            .then(col("value").diff(1, NullBehavior::Ignore))
            .otherwise(col("value"))])
        .sort(["group"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    let out = out.explode()?;
    assert_eq!(
        out,
        Series::new("value", &[0, 2, 1, 3, 2, 2, 7, 2, 3, 1, 2, 1])
    );
    Ok(())
}

#[test]
#[cfg(all(feature = "dtype-duration", feature = "dtype-decimal"))]
fn test_logical_mean_partitioned_group_by_block() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock();
    let df = df![
        "decimal" => [1, 1, 2],
        "duration" => [1000, 2000, 3000]
    ]?;

    let out = df
        .lazy()
        .with_column(col("decimal").cast(DataType::Decimal(None, Some(2))))
        .with_column(col("duration").cast(DataType::Duration(TimeUnit::Microseconds)))
        .group_by([col("decimal")])
        .agg([col("duration").mean()])
        .sort(["duration"], Default::default())
        .collect()?;

    let duration = out.column("duration")?;

    assert_eq!(
        duration.get(0)?,
        AnyValue::Duration(1500, TimeUnit::Microseconds)
    );

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
        AnyValue::List(Series::new("", [0, 1, 2, 3, 4]))
    );
    Ok(())
}
