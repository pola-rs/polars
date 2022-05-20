use super::*;
use polars_ops::prelude::ListNameSpaceImpl;

#[test]
fn test_agg_exprs() -> Result<()> {
    let df = fruits_cars();

    // a binary expression followed by a function and an aggregation. See if it runs
    let out = df
        .lazy()
        .groupby_stable([col("cars")])
        .agg([(lit(1) - col("A"))
            .map(|s| Ok(&s * 2), GetOutput::same_type())
            .list()
            .alias("foo")])
        .collect()?;
    let ca = out.column("foo")?.list()?;
    let out = ca.lst_lengths();

    assert_eq!(Vec::from(&out), &[Some(4), Some(1)]);
    Ok(())
}

#[test]
fn test_agg_unique_first() -> Result<()> {
    let df = df![
        "g"=> [1, 1, 2, 2, 3, 4, 1],
        "v"=> [1, 2, 2, 2, 3, 4, 1],
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("g")])
        .agg([
            col("v").unique().first().alias("v_first"),
            col("v").unique().sort(false).first().alias("true_first"),
            col("v").unique().list(),
        ])
        .collect()?;

    let a = out.column("v_first").unwrap();
    let a = a.sum::<i32>().unwrap();
    // can be both because unique does not guarantee order
    assert!(a == 10 || a == 11);

    let a = out.column("true_first").unwrap();
    let a = a.sum::<i32>().unwrap();
    // can be both because unique does not guarantee order
    assert_eq!(a, 10);

    Ok(())
}

#[test]
fn test_lazy_agg_scan() {
    let lf = scan_foods_csv;
    let df = lf().min().collect().unwrap();
    assert!(df.frame_equal_missing(&lf().collect().unwrap().min()));
    let df = lf().max().collect().unwrap();
    assert!(df.frame_equal_missing(&lf().collect().unwrap().max()));
    // mean is not yet aggregated at scan.
    let df = lf().mean().collect().unwrap();
    assert!(df.frame_equal_missing(&lf().collect().unwrap().mean()));
}

#[test]
fn test_lazy_df_aggregations() {
    let df = load_df();

    assert!(df
        .clone()
        .lazy()
        .min()
        .collect()
        .unwrap()
        .frame_equal_missing(&df.min()));
    assert!(df
        .clone()
        .lazy()
        .median()
        .collect()
        .unwrap()
        .frame_equal_missing(&df.median()));
    assert!(df
        .clone()
        .lazy()
        .quantile(0.5, QuantileInterpolOptions::default())
        .collect()
        .unwrap()
        .frame_equal_missing(
            &df.quantile(0.5, QuantileInterpolOptions::default())
                .unwrap()
        ));
}

#[test]
fn test_cumsum_agg_as_key() -> Result<()> {
    let df = df![
        "depth" => &[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "soil" => &["peat", "peat", "peat", "silt", "silt", "silt", "sand", "sand", "peat", "peat"]
    ]?;
    // this checks if the grouper can work with the complex query as a key

    let out = df
        .lazy()
        .groupby([col("soil")
            .neq(col("soil").shift_and_fill(1, col("soil").first()))
            .cumsum(false)
            .alias("key")])
        .agg([col("depth").max().keep_name()])
        .sort("depth", SortOptions::default())
        .collect()?;

    assert_eq!(
        Vec::from(out.column("key")?.u32()?),
        &[Some(0), Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        Vec::from(out.column("depth")?.i32()?),
        &[Some(2), Some(5), Some(7), Some(9)]
    );

    Ok(())
}

#[test]
#[cfg(feature = "moment")]
fn test_auto_skew_kurtosis_agg() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .clone()
        .lazy()
        .groupby([col("fruits")])
        .agg([
            col("B").skew(false).alias("bskew"),
            col("B").kurtosis(false, false).alias("bkurt"),
        ])
        .collect()?;

    assert!(matches!(out.column("bskew")?.dtype(), DataType::Float64));
    assert!(matches!(out.column("bkurt")?.dtype(), DataType::Float64));

    Ok(())
}

#[test]
fn test_auto_list_agg() -> Result<()> {
    let df = fruits_cars();

    // test if alias executor adds a list after shift and fill
    let out = df
        .clone()
        .lazy()
        .groupby([col("fruits")])
        .agg([col("B").shift_and_fill(-1, lit(-1)).alias("foo")])
        .collect()?;

    assert!(matches!(out.column("foo")?.dtype(), DataType::List(_)));

    // test if it runs and groupby executor thus implements a list after shift_and_fill
    let _out = df
        .clone()
        .lazy()
        .groupby([col("fruits")])
        .agg([col("B").shift_and_fill(-1, lit(-1))])
        .collect()?;

    // test if window expr executor adds list
    let _out = df
        .clone()
        .lazy()
        .select([col("B").shift_and_fill(-1, lit(-1)).alias("foo")])
        .collect()?;

    let _out = df
        .lazy()
        .select([col("B").shift_and_fill(-1, lit(-1))])
        .collect()?;
    Ok(())
}
#[test]
fn test_power_in_agg_list1() -> Result<()> {
    let df = fruits_cars();

    // this test if the group tuples are correctly updated after
    // a flat apply on a final aggregation
    let out = df
        .lazy()
        .groupby([col("fruits")])
        .agg([col("A")
            .rolling_min(RollingOptions {
                window_size: 1,
                ..Default::default()
            })
            .pow(2.0)
            .alias("foo")])
        .sort(
            "fruits",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    let agg = out.column("foo")?.list()?;
    let first = agg.get(0).unwrap();
    let vals = first.f64()?;
    assert_eq!(Vec::from(vals), &[Some(1.0), Some(4.0), Some(25.0)]);

    Ok(())
}

#[test]
fn test_power_in_agg_list2() -> Result<()> {
    let df = fruits_cars();

    // this test if the group tuples are correctly updated after
    // a flat apply on evaluate_on_groups
    let out = df
        .lazy()
        .groupby([col("fruits")])
        .agg([col("A")
            .rolling_min(RollingOptions {
                window_size: 2,
                min_periods: 2,
                ..Default::default()
            })
            .pow(2.0)
            .sum()
            .alias("foo")])
        .sort(
            "fruits",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    let agg = out.column("foo")?.f64()?;
    assert_eq!(Vec::from(agg), &[Some(5.0), Some(9.0)]);

    Ok(())
}
#[test]
fn test_binary_agg_context_0() -> Result<()> {
    let df = df![
        "groups" => [1, 1, 2, 2, 3, 3],
        "vals" => [1, 2, 3, 4, 5, 6]
    ]
    .unwrap();

    let out = df
        .lazy()
        .groupby_stable([col("groups")])
        .agg([when(col("vals").first().neq(lit(1)))
            .then(repeat("a", count()))
            .otherwise(repeat("b", count()))
            .alias("foo")])
        .collect()
        .unwrap();

    let out = out.column("foo")?;
    let out = out.explode()?;
    let out = out.utf8()?;
    assert_eq!(
        Vec::from(out),
        &[
            Some("b"),
            Some("b"),
            Some("a"),
            Some("a"),
            Some("a"),
            Some("a")
        ]
    );
    Ok(())
}

// just like binary expression, this must be changed. This can work
#[test]
fn test_binary_agg_context_1() -> Result<()> {
    let df = df![
        "groups" => [1, 1, 2, 2, 3, 3],
        "vals" => [1, 13, 3, 87, 1, 6]
    ]?;

    // groups
    // 1 => [1, 13]
    // 2 => [3, 87]
    // 3 => [1, 6]

    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("groups")])
        .agg([when(col("vals").eq(lit(1)))
            .then(col("vals").sum())
            .otherwise(lit(90))
            .alias("vals")])
        .collect()?;
    dbg!(&out);

    // if vals == 1 then sum(vals) else vals
    // [14, 90]
    // [90, 90]
    // [7, 90]
    let out = out.column("vals")?;
    let out = out.explode()?;
    let out = out.i32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(14), Some(90), Some(90), Some(90), Some(7), Some(90)]
    );

    let out = df
        .lazy()
        .groupby_stable([col("groups")])
        .agg([when(col("vals").eq(lit(1)))
            .then(lit(90))
            .otherwise(col("vals").sum())
            .alias("vals")])
        .collect()?;

    // if vals == 1 then 90 else sum(vals)
    // [90, 14]
    // [90, 90]
    // [90, 7]
    let out = out.column("vals")?;
    let out = out.explode()?;
    let out = out.i32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(90), Some(14), Some(90), Some(90), Some(90), Some(7)]
    );

    Ok(())
}

#[test]
fn test_binary_agg_context_2() -> Result<()> {
    let df = df![
        "groups" => [1, 1, 2, 2, 3, 3],
        "vals" => [1, 2, 3, 4, 5, 6]
    ]?;

    // this is complex because we first aggregate one expression of the binary operation.

    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("groups")])
        .agg([((col("vals").first() - col("vals")).list()).alias("vals")])
        .collect()?;

    // 0 - [1, 2] = [0, -1]
    // 3 - [3, 4] = [0, -1]
    // 5 - [5, 6] = [0, -1]
    let out = out.column("vals")?;
    let out = out.explode()?;
    let out = out.i32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(0), Some(-1), Some(0), Some(-1), Some(0), Some(-1)]
    );

    // Same, but now we reverse the lhs / rhs.
    let out = df
        .lazy()
        .groupby_stable([col("groups")])
        .agg([((col("vals")) - col("vals").first()).list().alias("vals")])
        .collect()?;

    // [1, 2] - 1 = [0, 1]
    // [3, 4] - 3 = [0, 1]
    // [5, 6] - 5 = [0, 1]
    let out = out.column("vals")?;
    let out = out.explode()?;
    let out = out.i32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(0), Some(1), Some(0), Some(1), Some(0), Some(1)]
    );

    Ok(())
}

#[test]
fn test_binary_agg_context_3() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .groupby_stable([col("cars")])
        .agg([(col("A") - col("A").first()).last().alias("last")])
        .collect()?;

    let out = out.column("last")?;
    assert_eq!(out.get(0), AnyValue::Int32(4));
    assert_eq!(out.get(1), AnyValue::Int32(0));

    Ok(())
}

#[test]
fn test_shift_elementwise_issue_2509() -> Result<()> {
    let df = df![
        "x"=> [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "y"=> [0, 10, 20, 0, 10, 20, 0, 10, 20]
    ]?;
    let out = df
        .lazy()
        // Don't use maintain order here! That hides the bug
        .groupby([col("x")])
        .agg(&[(col("y").shift(-1) + col("x")).list().alias("sum")])
        .sort("x", Default::default())
        .collect()?;

    let out = out.explode(["sum"])?;
    let out = out.column("sum")?;
    assert_eq!(out.get(0), AnyValue::Int32(10));
    assert_eq!(out.get(1), AnyValue::Int32(20));
    assert_eq!(out.get(2), AnyValue::Null);
    assert_eq!(out.get(3), AnyValue::Int32(11));
    assert_eq!(out.get(4), AnyValue::Int32(21));
    assert_eq!(out.get(5), AnyValue::Null);

    Ok(())
}

#[test]
fn take_aggregations() -> Result<()> {
    let df = df![
        "user" => ["lucy", "bob", "bob", "lucy", "tim"],
        "book" => ["c", "b", "a", "a", "a"],
        "count" => [3, 1, 2, 1, 1]
    ]?;

    let out = df
        .clone()
        .lazy()
        .groupby([col("user")])
        .agg([col("book").take(col("count").arg_max()).alias("fav_book")])
        .sort("user", Default::default())
        .collect()?;

    let s = out.column("fav_book")?;
    assert_eq!(s.get(0), AnyValue::Utf8("a"));
    assert_eq!(s.get(1), AnyValue::Utf8("c"));
    assert_eq!(s.get(2), AnyValue::Utf8("a"));

    let out = df
        .clone()
        .lazy()
        .groupby([col("user")])
        .agg([
            // keep the head as it test slice correctness
            col("book")
                .take(col("count").arg_sort(true).head(Some(2)))
                .alias("ordered"),
        ])
        .sort("user", Default::default())
        .collect()?;
    let s = out.column("ordered")?;
    let flat = s.explode()?;
    let flat = flat.utf8()?;
    let vals = flat.into_no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, ["a", "b", "c", "a", "a"]);

    let out = df
        .lazy()
        .groupby([col("user")])
        .agg([col("book").take(lit(0)).alias("take_lit")])
        .sort("user", Default::default())
        .collect()?;

    let taken = out.column("take_lit")?;
    let taken = taken.utf8()?;
    let vals = taken.into_no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, ["b", "c", "a"]);

    Ok(())
}
#[test]
fn test_take_consistency() -> Result<()> {
    let df = fruits_cars();
    let out = df
        .clone()
        .lazy()
        .select([col("A").arg_sort(true).take(lit(0))])
        .collect()?;

    let a = out.column("A")?;
    let a = a.idx()?;
    assert_eq!(a.get(0), Some(4));

    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("cars")])
        .agg([col("A").arg_sort(true).take(lit(0))])
        .collect()?;

    let out = out.column("A")?;
    let out = out.idx()?;
    assert_eq!(Vec::from(out), &[Some(3), Some(0)]);

    let out_df = df
        .clone()
        .lazy()
        .groupby_stable([col("cars")])
        .agg([
            col("A"),
            col("A").arg_sort(true).take(lit(0)).alias("1"),
            col("A")
                .take(col("A").arg_sort(true).take(lit(0)))
                .alias("2"),
        ])
        .collect()?;

    let out = out_df.column("2")?;
    let out = out.i32()?;
    assert_eq!(Vec::from(out), &[Some(5), Some(2)]);

    let out = out_df.column("1")?;
    let out = out.idx()?;
    assert_eq!(Vec::from(out), &[Some(3), Some(0)]);

    Ok(())
}

#[test]
fn test_take_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("fruits", Default::default())
        .select([col("B")
            .take(lit(Series::new("", &[0u32])))
            .over([col("fruits")])
            .alias("taken")])
        .collect()?;

    dbg!(&out);
    assert_eq!(
        Vec::from(out.column("taken")?.i32()?),
        &[Some(3), Some(3), Some(5), Some(5), Some(5)]
    );
    Ok(())
}
