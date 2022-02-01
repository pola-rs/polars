use super::*;

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
            col("v").unique().first(),
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
        .sort("depth", false)
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
        .sort("fruits", true)
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
        .sort("fruits", true)
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
            .then(lit("a"))
            .otherwise(lit("b"))
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
