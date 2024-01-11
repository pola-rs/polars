use chrono::NaiveDate;
use polars_ops::prelude::ListNameSpaceImpl;
use polars_utils::idxvec;

use super::*;

#[test]
#[cfg(feature = "dtype-datetime")]
fn test_agg_list_type() -> PolarsResult<()> {
    let s = Series::new("foo", &[1, 2, 3]);
    let s = s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?;

    let l = unsafe { s.agg_list(&GroupsProxy::Idx(vec![(0, idxvec![0, 1, 2])].into())) };

    let result = match l.dtype() {
        DataType::List(inner) => {
            matches!(&**inner, DataType::Datetime(TimeUnit::Nanoseconds, None))
        },
        _ => false,
    };
    assert!(result);

    Ok(())
}

#[test]
fn test_agg_exprs() -> PolarsResult<()> {
    let df = fruits_cars();

    // a binary expression followed by a function and an aggregation. See if it runs
    let out = df
        .lazy()
        .group_by_stable([col("cars")])
        .agg([(lit(1) - col("A"))
            .map(|s| Ok(Some(&s * 2)), GetOutput::same_type())
            .alias("foo")])
        .collect()?;
    let ca = out.column("foo")?.list()?;
    let out = ca.lst_lengths();

    assert_eq!(Vec::from(&out), &[Some(4), Some(1)]);
    Ok(())
}

#[test]
fn test_agg_unique_first() -> PolarsResult<()> {
    let df = df![
        "g"=> [1, 1, 2, 2, 3, 4, 1],
        "v"=> [1, 2, 2, 2, 3, 4, 1],
    ]?;

    let out = df
        .lazy()
        .group_by_stable([col("g")])
        .agg([
            col("v").unique().first().alias("v_first"),
            col("v").unique().sort(false).first().alias("true_first"),
            col("v").unique().implode(),
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
fn test_cum_sum_agg_as_key() -> PolarsResult<()> {
    let df = df![
        "depth" => &[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "soil" => &["peat", "peat", "peat", "silt", "silt", "silt", "sand", "sand", "peat", "peat"]
    ]?;
    // this checks if the grouper can work with the complex query as a key

    let out = df
        .lazy()
        .group_by([col("soil")
            .neq(col("soil").shift_and_fill(lit(1), col("soil").first()))
            .cum_sum(false)
            .alias("key")])
        .agg([col("depth").max().name().keep()])
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
fn test_auto_skew_kurtosis_agg() -> PolarsResult<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .group_by([col("fruits")])
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
fn test_auto_list_agg() -> PolarsResult<()> {
    let df = fruits_cars();

    // test if alias executor adds a list after shift and fill
    let out = df
        .clone()
        .lazy()
        .group_by([col("fruits")])
        .agg([col("B").shift_and_fill(lit(-1), lit(-1)).alias("foo")])
        .collect()?;

    assert!(matches!(out.column("foo")?.dtype(), DataType::List(_)));

    // test if it runs and group_by executor thus implements a list after shift_and_fill
    let _out = df
        .clone()
        .lazy()
        .group_by([col("fruits")])
        .agg([col("B").shift_and_fill(lit(-1), lit(-1))])
        .collect()?;

    // test if window expr executor adds list
    let _out = df
        .clone()
        .lazy()
        .select([col("B").shift_and_fill(lit(-1), lit(-1)).alias("foo")])
        .collect()?;

    let _out = df
        .lazy()
        .select([col("B").shift_and_fill(lit(-1), lit(-1))])
        .collect()?;
    Ok(())
}
#[test]
#[cfg(feature = "rolling_window")]
fn test_power_in_agg_list1() -> PolarsResult<()> {
    let df = fruits_cars();

    // this test if the group tuples are correctly updated after
    // a flat apply on a final aggregation
    let out = df
        .lazy()
        .group_by([col("fruits")])
        .agg([
            col("A")
                .rolling_min(RollingOptions {
                    window_size: Duration::new(1),
                    ..Default::default()
                })
                .alias("input"),
            col("A")
                .rolling_min(RollingOptions {
                    window_size: Duration::new(1),
                    ..Default::default()
                })
                .pow(2.0)
                .alias("foo"),
        ])
        .sort(
            "fruits",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    let agg = out.column("foo")?.list()?;
    let first = agg.get_as_series(0).unwrap();
    let vals = first.f64()?;
    assert_eq!(Vec::from(vals), &[Some(1.0), Some(4.0), Some(25.0)]);

    Ok(())
}

#[test]
#[cfg(feature = "rolling_window")]
fn test_power_in_agg_list2() -> PolarsResult<()> {
    let df = fruits_cars();

    // this test if the group tuples are correctly updated after
    // a flat apply on evaluate_on_groups
    let out = df
        .lazy()
        .group_by([col("fruits")])
        .agg([col("A")
            .rolling_min(RollingOptions {
                window_size: Duration::new(2),
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
fn test_binary_agg_context_0() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 1, 2, 2, 3, 3],
        "vals" => [1, 2, 3, 4, 5, 6]
    ]
    .unwrap();

    let out = df
        .lazy()
        .group_by_stable([col("groups")])
        .agg([when(col("vals").first().neq(lit(1)))
            .then(repeat(lit("a"), count()))
            .otherwise(repeat(lit("b"), count()))
            .alias("foo")])
        .collect()
        .unwrap();

    let out = out.column("foo")?;
    let out = out.explode()?;
    let out = out.str()?;
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
fn test_binary_agg_context_1() -> PolarsResult<()> {
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
        .group_by_stable([col("groups")])
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
        .group_by_stable([col("groups")])
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
fn test_binary_agg_context_2() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 1, 2, 2, 3, 3],
        "vals" => [1, 2, 3, 4, 5, 6]
    ]?;

    // this is complex because we first aggregate one expression of the binary operation.

    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("groups")])
        .agg([(col("vals").first() - col("vals")).alias("vals")])
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
        .group_by_stable([col("groups")])
        .agg([((col("vals")) - col("vals").first()).alias("vals")])
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
fn test_binary_agg_context_3() -> PolarsResult<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .group_by_stable([col("cars")])
        .agg([(col("A") - col("A").first()).last().alias("last")])
        .collect()?;

    let out = out.column("last")?;
    assert_eq!(out.get(0)?, AnyValue::Int32(4));
    assert_eq!(out.get(1)?, AnyValue::Int32(0));

    Ok(())
}

#[test]
fn test_shift_elementwise_issue_2509() -> PolarsResult<()> {
    let df = df![
        "x"=> [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "y"=> [0, 10, 20, 0, 10, 20, 0, 10, 20]
    ]?;
    let out = df
        .lazy()
        // Don't use maintain order here! That hides the bug
        .group_by([col("x")])
        .agg(&[(col("y").shift(lit(-1)) + col("x")).alias("sum")])
        .sort("x", Default::default())
        .collect()?;

    let out = out.explode(["sum"])?;
    let out = out.column("sum")?;
    assert_eq!(out.get(0)?, AnyValue::Int32(10));
    assert_eq!(out.get(1)?, AnyValue::Int32(20));
    assert_eq!(out.get(2)?, AnyValue::Null);
    assert_eq!(out.get(3)?, AnyValue::Int32(11));
    assert_eq!(out.get(4)?, AnyValue::Int32(21));
    assert_eq!(out.get(5)?, AnyValue::Null);

    Ok(())
}

#[test]
fn take_aggregations() -> PolarsResult<()> {
    let df = df![
        "user" => ["lucy", "bob", "bob", "lucy", "tim"],
        "book" => ["c", "b", "a", "a", "a"],
        "count" => [3, 1, 2, 1, 1]
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by([col("user")])
        .agg([col("book").get(col("count").arg_max()).alias("fav_book")])
        .sort("user", Default::default())
        .collect()?;

    let s = out.column("fav_book")?;
    assert_eq!(s.get(0)?, AnyValue::String("a"));
    assert_eq!(s.get(1)?, AnyValue::String("c"));
    assert_eq!(s.get(2)?, AnyValue::String("a"));

    let out = df
        .clone()
        .lazy()
        .group_by([col("user")])
        .agg([
            // keep the head as it test slice correctness
            col("book")
                .gather(
                    col("count")
                        .arg_sort(SortOptions {
                            descending: true,
                            nulls_last: false,
                            multithreaded: true,
                            maintain_order: false,
                        })
                        .head(Some(2)),
                )
                .alias("ordered"),
        ])
        .sort("user", Default::default())
        .collect()?;
    let s = out.column("ordered")?;
    let flat = s.explode()?;
    let flat = flat.str()?;
    let vals = flat.into_no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, ["a", "b", "c", "a", "a"]);

    let out = df
        .lazy()
        .group_by([col("user")])
        .agg([col("book").get(lit(0)).alias("take_lit")])
        .sort("user", Default::default())
        .collect()?;

    let taken = out.column("take_lit")?;
    let taken = taken.str()?;
    let vals = taken.into_no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, ["b", "c", "a"]);

    Ok(())
}
#[test]
fn test_take_consistency() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .clone()
        .lazy()
        .select([col("A")
            .arg_sort(SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: true,
                maintain_order: false,
            })
            .get(lit(0))])
        .collect()?;

    let a = out.column("A")?;
    let a = a.idx()?;
    assert_eq!(a.get(0), Some(4));

    let out = df
        .clone()
        .lazy()
        .group_by_stable([col("cars")])
        .agg([col("A")
            .arg_sort(SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: true,
                maintain_order: false,
            })
            .get(lit(0))])
        .collect()?;

    let out = out.column("A")?;
    let out = out.idx()?;
    assert_eq!(Vec::from(out), &[Some(3), Some(0)]);

    let out_df = df
        .lazy()
        .group_by_stable([col("cars")])
        .agg([
            col("A"),
            col("A")
                .arg_sort(SortOptions {
                    descending: true,
                    nulls_last: false,
                    multithreaded: true,
                    maintain_order: false,
                })
                .get(lit(0))
                .alias("1"),
            col("A")
                .get(
                    col("A")
                        .arg_sort(SortOptions {
                            descending: true,
                            nulls_last: false,
                            multithreaded: true,
                            maintain_order: false,
                        })
                        .get(lit(0)),
                )
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
fn test_take_in_groups() -> PolarsResult<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("fruits", Default::default())
        .select([col("B").get(lit(0u32)).over([col("fruits")]).alias("taken")])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("taken")?.i32()?),
        &[Some(3), Some(3), Some(5), Some(5), Some(5)]
    );
    Ok(())
}

#[test]
fn test_describe() -> PolarsResult<()> {
    std::env::set_var("POLARS_FMT_MAX_COLS", "100"); //FMT_MAX_COLS is pub(crate) in polars_core :(
    std::env::set_var("POLARS_FMT_MAX_ROWS", "100");

    let df = all_types_df();

    let summary_df = df.lazy().describe()?;

    let desc_cl = summary_df
        .column("describe")?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect::<Vec<&str>>();
    let float_cl = summary_df
        .column("float")?
        .f64()?
        .into_iter()
        .map(|v| format!("{:.5}", v.unwrap_or_default()))
        .collect::<Vec<String>>();
    let int_cl = summary_df
        .column("int")?
        .f64()?
        .into_iter()
        .map(|v| format!("{:.5}", v.unwrap_or_default()))
        .collect::<Vec<String>>();
    let bool_cl = summary_df
        .column("bool")?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect::<Vec<&str>>();
    let str_cl = summary_df
        .column("str")?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect::<Vec<&str>>();
    let str2_cl = summary_df
        .column("str2")?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect::<Vec<&str>>();
    let date_cl = summary_df
        .column("date")?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect::<Vec<&str>>();

    /*      from py-polars/polars/dataframe/frame.py:4394
           ┌────────────┬──────────┬──────────┬───────┬──────┬──────┬────────────┐
           │ describe   ┆ float    ┆ int      ┆ bool  ┆ str  ┆ str2 ┆ date      │
           │ ---        ┆ ---      ┆ ---      ┆ ---   ┆ ---  ┆ ---  ┆ ---       │
           │ str        ┆ f64      ┆ f64      ┆ str   ┆ str  ┆ str  ┆ str       │
           ╞════════════╪══════════╪══════════╪═══════╪══════╪══════╪════════════╡
           │ count      ┆ 3.0      ┆ 2.0      ┆ 3     ┆ 2    ┆ 2    ┆ 3         │
           │ null_count ┆ 0.0      ┆ 1.0      ┆ 0     ┆ 1    ┆ 1    ┆ 0         │
           │ mean       ┆ 2.266667 ┆ 4.5      ┆ null  ┆ null ┆ null ┆ null      │
           │ std        ┆ 1.101514 ┆ 0.707107 ┆ null  ┆ null ┆ null ┆ null      │
           │ min        ┆ 1.0      ┆ 4.0      ┆ False ┆ b    ┆ eur  ┆ 2020-01-01│
           │ 25%        ┆ 2.8      ┆ 4.0      ┆ null  ┆ null ┆ null ┆ null      │
           │ 50%        ┆ 2.8      ┆ 5.0      ┆ null  ┆ null ┆ null ┆ null      │
           │ 75%        ┆ 3.0      ┆ 5.0      ┆ null  ┆ null ┆ null ┆ null      │
           │ max        ┆ 3.0      ┆ 5.0      ┆ True  ┆ c    ┆ usd  ┆ 2022-01-01│
           └────────────┴──────────┴──────────┴───────┴──────┴──────┴────────────┘
    */

    assert_eq!(desc_cl[0], "count");
    assert_eq!(float_cl[0], "3.00000");
    assert_eq!(int_cl[0], "2.00000");
    assert_eq!(bool_cl[0], "3");
    assert_eq!(desc_cl[1], "null_count");
    assert_eq!(float_cl[1], "0.00000");
    assert_eq!(int_cl[1], "1.00000");
    assert_eq!(bool_cl[1], "0");
    assert_eq!(desc_cl[2], "mean");
    assert_eq!(float_cl[2], "2.26667");
    assert_eq!(int_cl[2], "4.50000");
    assert_eq!(bool_cl[2], "");
    assert_eq!(desc_cl[3], "std");
    assert_eq!(float_cl[3], "1.10151");
    assert_eq!(int_cl[3], "0.70711");
    assert_eq!(bool_cl[3], "");
    assert_eq!(desc_cl[4], "min");
    assert_eq!(float_cl[4], "1.00000");
    assert_eq!(int_cl[4], "4.00000");
    assert_eq!(bool_cl[4], "false");
    assert_eq!(desc_cl[5], "25%");
    assert_eq!(float_cl[5], "2.80000");
    assert_eq!(int_cl[5], "4.00000");
    assert_eq!(bool_cl[5], "");
    assert_eq!(desc_cl[6], "50%");
    assert_eq!(float_cl[6], "2.80000");
    assert_eq!(int_cl[6], "5.00000");
    assert_eq!(bool_cl[6], "");
    assert_eq!(desc_cl[7], "75%");
    assert_eq!(float_cl[7], "3.00000");
    assert_eq!(int_cl[7], "5.00000");
    assert_eq!(bool_cl[7], "");
    assert_eq!(desc_cl[8], "max");
    assert_eq!(float_cl[8], "3.00000");
    assert_eq!(int_cl[8], "5.00000");
    assert_eq!(bool_cl[8], "true");

    assert_eq!(str_cl[0], "2");
    assert_eq!(str2_cl[0], "2");
    assert_eq!(date_cl[0], "3");
    assert_eq!(str_cl[1], "1");
    assert_eq!(str2_cl[1], "1");
    assert_eq!(date_cl[1], "0");
    assert_eq!(str_cl[2], "");
    assert_eq!(str2_cl[2], "");
    assert_eq!(date_cl[2], "");
    assert_eq!(str_cl[3], "");
    assert_eq!(str2_cl[3], "");
    assert_eq!(date_cl[3], "");
    assert_eq!(str_cl[4], "b");
    assert_eq!(str2_cl[4], "eur");
    assert_eq!(date_cl[4], "2020-01-01");
    assert_eq!(str_cl[5], "");
    assert_eq!(str2_cl[5], "");
    assert_eq!(date_cl[5], "");
    assert_eq!(str_cl[6], "");
    assert_eq!(str2_cl[6], "");
    assert_eq!(date_cl[6], "");
    assert_eq!(str_cl[7], "");
    assert_eq!(str2_cl[7], "");
    assert_eq!(date_cl[7], "");
    assert_eq!(str_cl[8], "c");
    assert_eq!(str2_cl[8], "usd");
    assert_eq!(date_cl[8], "2022-01-01");

    Ok(())
}

#[test]
#[cfg(feature = "pivot")]
fn test_describe_with_extra_aggs() -> PolarsResult<()> {
    std::env::set_var("POLARS_FMT_MAX_COLS", "100");
    std::env::set_var("POLARS_FMT_MAX_ROWS", "100");

    let df = all_types_df();

    let summary_df = df.lazy().describe_with_params(
        false,
        vec![(
            "kurtosis".to_owned(),
            Box::new(|dt: &DataType| dt.is_numeric()) as Box<dyn Fn(&DataType) -> bool>,
            Box::new(move |name: &String| {
                col(name)
                    .kurtosis(true, true)
                    .alias(format!("{} kurtosis", name).as_str())
            }) as Box<dyn Fn(&String) -> Expr>,
        )],
        Some(vec!["float"]),
    )?;

    assert_eq!(
        summary_df
            .get_columns()
            .iter()
            .map(|s| s.name())
            .collect::<Vec<&str>>(),
        vec!["name", "kurtosis"]
    );

    assert_eq!(
        summary_df
            .column("name")?
            .str()?
            .into_iter()
            .collect::<Vec<Option<&str>>>()[0],
        Some("float"),
    );

    let kurtosis_cl = summary_df
        .column("kurtosis")?
        .f64()?
        .into_iter()
        .map(|v| format!("{:.5}", v.unwrap_or_default()))
        .collect::<Vec<String>>();

    assert_eq!(kurtosis_cl, ["-1.50000"]);

    Ok(())
}

#[test]
#[cfg(feature = "pivot")]
fn test_describe_nan() -> PolarsResult<()> {
    std::env::set_var("POLARS_FMT_MAX_COLS", "100");
    std::env::set_var("POLARS_FMT_MAX_ROWS", "100");

    let df = DataFrame::new(vec![Series::new("days", [f32::NAN].as_ref())])?;

    let summary_df = df.lazy().describe()?;

    assert_eq!(
        summary_df
            .column("days")?
            .f64()?
            .into_iter()
            .collect::<Vec<Option<f64>>>()[0..2],
        vec![
            Some(1.0),
            Some(0.0),
            Some(f64::NAN),
            None,
            Some(f64::NAN),
            Some(f64::NAN),
            Some(f64::NAN),
            Some(f64::NAN),
            Some(f64::NAN)
        ][0..2]
    );

    Ok(())
}

fn all_types_df() -> DataFrame {
    df![
        "float" => [Some(1.0), Some(2.8), Some(3.0)],
        "int" =>  [Some(4), Some(5), None],
        "bool" =>  [true, false, true],
        "str" =>  [None, Some("b"), Some("c")],
        "str2" =>  [Some("usd"), Some("eur"), None],
        "date" =>  [NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2022, 1, 1).unwrap()],
    ].unwrap()
}
