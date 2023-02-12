use polars_core::frame::explode::MeltArgs;
use polars_core::series::ops::NullBehavior;

use super::*;

#[test]
fn test_lazy_with_column() {
    let df = get_df()
        .lazy()
        .with_column(lit(10).alias("foo"))
        .collect()
        .unwrap();
    assert_eq!(df.width(), 6);
    assert!(df.column("foo").is_ok());
}

#[test]
fn test_lazy_exec() {
    let df = get_df();
    let _new = df
        .clone()
        .lazy()
        .select([col("sepal.width"), col("variety")])
        .sort("sepal.width", Default::default())
        .collect();

    let new = df
        .lazy()
        .filter(not(col("sepal.width").lt(lit(3.5))))
        .collect()
        .unwrap();

    let check = new.column("sepal.width").unwrap().f64().unwrap().gt(3.4);
    assert!(check.all())
}

#[test]
fn test_lazy_alias() {
    let df = get_df();
    let new = df
        .lazy()
        .select([col("sepal.width").alias("petals"), col("sepal.width")])
        .collect()
        .unwrap();
    assert_eq!(new.get_column_names(), &["petals", "sepal.width"]);
}

#[test]
fn test_lazy_melt() {
    let df = get_df();

    let args = MeltArgs {
        id_vars: vec!["petal.width".to_string(), "petal.length".to_string()],
        value_vars: vec!["sepal.length".to_string(), "sepal.width".to_string()],
        variable_name: None,
        value_name: None,
    };

    let out = df
        .lazy()
        .melt(args)
        .filter(col("variable").eq(lit("sepal.length")))
        .select([col("variable"), col("petal.width"), col("value")])
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (7, 3));
}

#[test]
fn test_lazy_drop_nulls() {
    let df = df! {
        "foo" => &[Some(1), None, Some(3)],
        "bar" => &[Some(1), Some(2), None]
    }
    .unwrap();

    let new = df.lazy().drop_nulls(None).collect().unwrap();
    let out = df! {
        "foo" => &[Some(1)],
        "bar" => &[Some(1)]
    }
    .unwrap();
    assert!(new.frame_equal(&out));
}

#[test]
fn test_lazy_udf() {
    let df = get_df();
    let new = df
        .lazy()
        .select([col("sepal.width").map(|s| Ok(Some(s * 200.0)), GetOutput::same_type())])
        .collect()
        .unwrap();
    assert_eq!(
        new.column("sepal.width").unwrap().f64().unwrap().get(0),
        Some(700.0)
    );
}

#[test]
fn test_lazy_is_null() {
    let df = get_df();
    let new = df
        .clone()
        .lazy()
        .filter(col("sepal.width").is_null())
        .collect()
        .unwrap();

    assert_eq!(new.height(), 0);

    let new = df
        .clone()
        .lazy()
        .filter(col("sepal.width").is_not_null())
        .collect()
        .unwrap();
    assert_eq!(new.height(), df.height());

    let new = df
        .lazy()
        .groupby([col("variety")])
        .agg([col("sepal.width").min()])
        .collect()
        .unwrap();

    assert_eq!(new.shape(), (1, 2));
}

#[test]
fn test_lazy_pushdown_through_agg() {
    // An aggregation changes the schema names, check if the pushdown succeeds.
    let df = get_df();
    let new = df
        .lazy()
        .groupby([col("variety")])
        .agg([
            col("sepal.length").min(),
            col("petal.length").min().alias("foo"),
        ])
        .select([col("foo")])
        // second selection is to test if optimizer can handle that
        .select([col("foo").alias("bar")])
        .collect()
        .unwrap();

    assert_eq!(new.shape(), (1, 1));
    let bar = new.column("bar").unwrap();
    assert_eq!(bar.get(0).unwrap(), AnyValue::Float64(1.3));
}

#[test]
fn test_lazy_shift() {
    let df = get_df();
    let new = df
        .lazy()
        .select([col("sepal.width").alias("foo").shift(2)])
        .collect()
        .unwrap();
    assert_eq!(new.column("foo").unwrap().f64().unwrap().get(0), None);
}

#[test]
fn test_shift_and_fill() -> PolarsResult<()> {
    let out = df![
        "a" => [1, 2, 3]
    ]?
    .lazy()
    .select([col("a").shift_and_fill(-1, lit(5))])
    .collect()?;

    let out = out.column("a")?;
    assert_eq!(Vec::from(out.i32()?), &[Some(2), Some(3), Some(5)]);
    Ok(())
}

#[test]
fn test_lazy_ternary_and_predicates() {
    let df = get_df();
    // test if this runs. This failed because is_not_null changes the schema name, so we
    // really need to check the root column
    let ldf = df
        .clone()
        .lazy()
        .with_column(lit(3).alias("foo"))
        .filter(col("foo").is_not_null());
    let _new = ldf.collect().unwrap();

    let ldf = df
        .lazy()
        .with_column(
            when(col("sepal.length").lt(lit(5.0)))
                .then(
                    lit(3), // is another type on purpose to check type coercion
                )
                .otherwise(col("sepal.width"))
                .alias("foo"),
        )
        .filter(col("foo").gt(lit(3.0)));

    let new = ldf.collect().unwrap();
    let length = new.column("sepal.length").unwrap();
    assert_eq!(length, &Series::new("sepal.length", &[5.1f64, 5.0, 5.4]));
    assert_eq!(new.shape(), (3, 6));
}

#[test]
fn test_lazy_binary_ops() {
    let df = df!("a" => &[1, 2, 3, 4, 5, ]).unwrap();
    let new = df
        .lazy()
        .select([col("a").eq(lit(2)).alias("foo")])
        .collect()
        .unwrap();
    assert_eq!(new.column("foo").unwrap().sum::<i32>(), Some(1));
}

#[test]
fn test_lazy_query_2() {
    let df = load_df();
    let ldf = df
        .lazy()
        .with_column(col("a").map(|s| Ok(Some(s * 2)), GetOutput::same_type()))
        .filter(col("a").lt(lit(2)))
        .select([col("b"), col("a")]);

    let new = ldf.collect().unwrap();
    assert_eq!(new.shape(), (0, 2));
}

#[test]
fn test_lazy_query_3() {
    // query checks if schema of scanning is not changed by aggregation
    let _ = scan_foods_csv()
        .groupby([col("calories")])
        .agg([col("fats_g").max()])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_query_4() {
    let df = df! {
        "uid" => [0, 0, 0, 1, 1, 1],
        "day" => [1, 2, 3, 1, 2, 3],
        "cumcases" => [10, 12, 15, 25, 30, 41]
    }
    .unwrap();

    let base_df = df.lazy();

    let out = base_df
        .clone()
        .groupby([col("uid")])
        .agg([
            col("day").alias("day"),
            col("cumcases")
                .apply(
                    |s: Series| Ok(Some(&s - &(s.shift(1)))),
                    GetOutput::same_type(),
                )
                .alias("diff_cases"),
        ])
        .explode([col("day"), col("diff_cases")])
        .join(
            base_df,
            [col("uid"), col("day")],
            [col("uid"), col("day")],
            JoinType::Inner,
        )
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("diff_cases").unwrap().i32().unwrap()),
        &[None, Some(2), Some(3), None, Some(5), Some(11)]
    );
}

#[test]
fn test_lazy_query_5() {
    // if this one fails, the list builder probably does not handle offsets
    let df = df! {
        "uid" => [0, 0, 0, 1, 1, 1],
        "day" => [1, 2, 4, 1, 2, 3],
        "cumcases" => [10, 12, 15, 25, 30, 41]
    }
    .unwrap();

    let out = df
        .lazy()
        .groupby([col("uid")])
        .agg([col("day").head(Some(2))])
        .collect()
        .unwrap();
    let s = out
        .select_at_idx(1)
        .unwrap()
        .list()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(s.len(), 2);
    let s = out
        .select_at_idx(1)
        .unwrap()
        .list()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(s.len(), 2);
}

#[test]
#[cfg(feature = "is_in")]
fn test_lazy_query_8() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/842
    let df = df![
        "A" => [1, 2, 3],
        "B" => [1, 2, 3],
        "C" => [1, 2, 3],
        "D" => [1, 2, 3],
        "E" => [1, 2, 3]
    ]?;

    let mut selection = vec![];

    for c in &["A", "B", "C", "D", "E"] {
        let e = when(col(c).is_in(col("E")))
            .then(col("A"))
            .otherwise(Null {}.lit())
            .alias(c);
        selection.push(e);
    }

    let out = df
        .lazy()
        .select(selection)
        .filter(col("D").gt(lit(1)))
        .collect()?;
    assert_eq!(out.shape(), (2, 5));
    Ok(())
}

#[test]
fn test_lazy_query_9() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/958
    let cities = df![
        "Cities.City"=> ["Moscow", "Berlin", "Paris","Hamburg", "Lyon", "Novosibirsk"],
        "Cities.Population"=> [11.92, 3.645, 2.161, 1.841, 0.513, 1.511],
        "Cities.Country"=> ["Russia", "Germany", "France", "Germany", "France", "Russia"]
    ]?;

    let sales = df![
               "Sales.City"=> ["Moscow", "Berlin", "Paris", "Moscow", "Berlin", "Paris", "Moscow", "Berlin", "Paris"],
    "Sales.Item"=> ["Item A", "Item A","Item A",
                   "Item B", "Item B","Item B",
                   "Item C", "Item C","Item C"],
    "Sales.Amount"=> [200, 180, 100,
                    3, 30, 20,
                    90, 130, 125]
        ]?;

    let out = sales
        .lazy()
        .join(
            cities.lazy(),
            [col("Sales.City")],
            [col("Cities.City")],
            JoinType::Inner,
        )
        .groupby([col("Cities.Country")])
        .agg([col("Sales.Amount").sum().alias("sum")])
        .sort("sum", Default::default())
        .collect()?;
    let vals = out
        .column("sum")?
        .i32()?
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(vals, &[245, 293, 340]);
    Ok(())
}

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration"
))]
fn test_lazy_query_10() {
    use polars_core::export::chrono::Duration as ChronoDuration;
    let date = NaiveDate::from_ymd_opt(2021, 3, 5).unwrap();
    let x: Series = DatetimeChunked::from_naive_datetime(
        "x",
        [
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(13, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(14, 0, 0).unwrap()),
        ],
        TimeUnit::Nanoseconds,
    )
    .into();
    let y: Series = DatetimeChunked::from_naive_datetime(
        "y",
        [
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(11, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(11, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(11, 0, 0).unwrap()),
        ],
        TimeUnit::Nanoseconds,
    )
    .into();
    let df = DataFrame::new(vec![x, y]).unwrap();
    let out = df
        .lazy()
        .select(&[(col("x") - col("y")).alias("z")])
        .collect()
        .unwrap();
    let z: Series = DurationChunked::from_duration(
        "z",
        [
            ChronoDuration::hours(1),
            ChronoDuration::hours(2),
            ChronoDuration::hours(3),
        ],
        TimeUnit::Nanoseconds,
    )
    .into();
    assert!(out.column("z").unwrap().series_equal(&z));
    let x: Series = DatetimeChunked::from_naive_datetime(
        "x",
        [
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(2, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(3, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(4, 0, 0).unwrap()),
        ],
        TimeUnit::Milliseconds,
    )
    .into();
    let y: Series = DatetimeChunked::from_naive_datetime(
        "y",
        [
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(1, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(1, 0, 0).unwrap()),
            NaiveDateTime::new(date, NaiveTime::from_hms_opt(1, 0, 0).unwrap()),
        ],
        TimeUnit::Nanoseconds,
    )
    .into();
    let df = DataFrame::new(vec![x, y]).unwrap();
    let out = df
        .lazy()
        .select(&[(col("x") - col("y")).alias("z")])
        .collect()
        .unwrap();
    assert!(out
        .column("z")
        .unwrap()
        .series_equal(&z.cast(&DataType::Duration(TimeUnit::Milliseconds)).unwrap()));
}

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dtype-date",
    feature = "dtype-datetime"
))]
fn test_lazy_query_7() {
    let date = NaiveDate::from_ymd_opt(2021, 3, 5).unwrap();
    let dates = [
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 0, 0).unwrap()),
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 1, 0).unwrap()),
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 2, 0).unwrap()),
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 3, 0).unwrap()),
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 4, 0).unwrap()),
        NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 5, 0).unwrap()),
    ];
    let data = vec![Some(1.), Some(2.), Some(3.), Some(4.), None, None];
    let df = DataFrame::new(vec![
        DatetimeChunked::from_naive_datetime("date", dates, TimeUnit::Nanoseconds).into(),
        Series::new("data", data),
    ])
    .unwrap();
    // this tests if predicate pushdown not interferes with the shift data.
    let out = df
        .lazy()
        .with_column(col("data").shift(-1).alias("output"))
        .with_column(col("output").shift(2).alias("shifted"))
        .filter(col("date").gt(lit(NaiveDateTime::new(
            date,
            NaiveTime::from_hms_opt(12, 2, 0).unwrap(),
        ))))
        .collect()
        .unwrap();
    let a = out.column("shifted").unwrap().sum::<f64>().unwrap() - 7.0;
    assert!(a < 0.01 && a > -0.01);
}

#[test]
fn test_lazy_shift_and_fill_all() {
    let data = &[1, 2, 3];
    let df = DataFrame::new(vec![Series::new("data", data)]).unwrap();
    let out = df
        .lazy()
        .with_column(col("data").shift(1).fill_null(lit(0)).alias("output"))
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("output").unwrap().i32().unwrap()),
        vec![Some(0), Some(1), Some(2)]
    );
}

#[test]
fn test_lazy_shift_operation_no_filter() {
    // check if predicate pushdown optimization does not fail
    let df = df! {
        "a" => &[1, 2, 3],
        "b" => &[1, 2, 3]
    }
    .unwrap();
    df.lazy()
        .with_column(col("b").shift(1).alias("output"))
        .collect()
        .unwrap();
}

#[test]
fn test_simplify_expr() {
    // Test if expression containing literals is simplified
    let df = get_df();

    let plan = df
        .lazy()
        .select(&[lit(1.0f32) + lit(1.0f32) + col("sepal.width")])
        .logical_plan;

    let mut expr_arena = Arena::new();
    let mut lp_arena = Arena::new();
    let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(SimplifyExprRule {})];

    let optimizer = StackOptimizer {};
    let mut lp_top = to_alp(plan, &mut expr_arena, &mut lp_arena).unwrap();
    lp_top = optimizer
        .optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top)
        .unwrap();
    let plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
    assert!(
        matches!(plan, LogicalPlan::Projection{ expr, ..} if matches!(&expr[0], Expr::BinaryExpr{left, ..} if **left == Expr::Literal(LiteralValue::Float32(2.0))))
    );
}

#[test]
fn test_lazy_wildcard() {
    let df = load_df();
    let new = df.clone().lazy().select([col("*")]).collect().unwrap();
    assert_eq!(new.shape(), (5, 3));

    let new = df
        .lazy()
        .groupby([col("b")])
        .agg([col("*").sum().suffix(""), col("*").first().suffix("_first")])
        .collect()
        .unwrap();
    assert_eq!(new.shape(), (3, 5)); // Should exclude b from wildcard aggregations.
}

#[test]
fn test_lazy_reverse() {
    let df = load_df();
    assert!(df
        .clone()
        .lazy()
        .reverse()
        .collect()
        .unwrap()
        .frame_equal_missing(&df.reverse()))
}

#[test]
fn test_lazy_fill_null() {
    let df = df! {
        "a" => &[None, Some(2.0)],
        "b" => &[Some(1.0), None]
    }
    .unwrap();
    let out = df.lazy().fill_null(lit(10.0)).collect().unwrap();
    let correct = df! {
        "a" => &[Some(10.0), Some(2.0)],
        "b" => &[Some(1.0), Some(10.0)]
    }
    .unwrap();
    assert!(out.frame_equal(&correct));
    assert_eq!(out.get_column_names(), vec!["a", "b"])
}

#[test]
fn test_lazy_double_projection() {
    let df = df! {
        "foo" => &[1, 2, 3]
    }
    .unwrap();
    df.lazy()
        .select([col("foo").alias("bar")])
        .select([col("bar")])
        .collect()
        .unwrap();
}

#[test]
fn test_type_coercion() {
    let df = df! {
        "foo" => &[1, 2, 3],
        "bar" => &[1.0, 2.0, 3.0]
    }
    .unwrap();

    let lp = df.lazy().select([col("foo") * col("bar")]).logical_plan;

    let mut expr_arena = Arena::new();
    let mut lp_arena = Arena::new();
    let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(TypeCoercionRule {})];

    let optimizer = StackOptimizer {};
    let mut lp_top = to_alp(lp, &mut expr_arena, &mut lp_arena).unwrap();
    lp_top = optimizer
        .optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top)
        .unwrap();
    let lp = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);

    if let LogicalPlan::Projection { expr, .. } = lp {
        if let Expr::BinaryExpr { left, right, .. } = &expr[0] {
            assert!(matches!(&**left, Expr::Cast { .. }));
            // bar is already float, does not have to be coerced
            assert!(matches!(&**right, Expr::Column { .. }));
        } else {
            panic!()
        }
    };
}

#[test]
fn test_lazy_partition_agg() {
    let df = df! {
        "foo" => &[1, 1, 2, 2, 3],
        "bar" => &[1.0, 1.0, 2.0, 2.0, 3.0]
    }
    .unwrap();

    let out = df
        .lazy()
        .groupby([col("foo")])
        .agg([col("bar").mean()])
        .sort("foo", Default::default())
        .collect()
        .unwrap();

    assert_eq!(
        Vec::from(out.column("bar").unwrap().f64().unwrap()),
        &[Some(1.0), Some(2.0), Some(3.0)]
    );

    let out = scan_foods_csv()
        .groupby([col("category")])
        .agg([col("calories")])
        .sort("category", Default::default())
        .collect()
        .unwrap();
    let cat_agg_list = out.select_at_idx(1).unwrap();
    let fruit_series = cat_agg_list.list().unwrap().get(0).unwrap();
    let fruit_list = fruit_series.i64().unwrap();
    assert_eq!(
        Vec::from(fruit_list),
        &[
            Some(60),
            Some(30),
            Some(50),
            Some(30),
            Some(60),
            Some(130),
            Some(50),
        ]
    )
}

#[test]
fn test_lazy_groupby_apply() {
    let df = fruits_cars();

    df.lazy()
        .groupby([col("fruits")])
        .agg([col("cars").apply(
            |s: Series| Ok(Some(Series::new("", &[s.len() as u32]))),
            GetOutput::same_type(),
        )])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_shift_and_fill() {
    let df = df! {
        "A" => &[1, 2, 3, 4, 5],
        "B" => &[5, 4, 3, 2, 1]
    }
    .unwrap();
    let out = df
        .clone()
        .lazy()
        .with_column(col("A").shift_and_fill(2, col("B").mean()))
        .collect()
        .unwrap();
    assert_eq!(out.column("A").unwrap().null_count(), 0);

    // shift from the other side
    let out = df
        .clone()
        .lazy()
        .with_column(col("A").shift_and_fill(-2, col("B").mean()))
        .collect()
        .unwrap();
    assert_eq!(out.column("A").unwrap().null_count(), 0);

    let out = df
        .lazy()
        .shift_and_fill(-1, col("B").std(1))
        .collect()
        .unwrap();
    assert_eq!(out.column("A").unwrap().null_count(), 0);
}

#[test]
fn test_lazy_groupby() {
    let df = df! {
        "a" => &[Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)],
        "groups" => &["a", "a", "b", "c", "c"]
    }
    .unwrap();

    let out = df
        .lazy()
        .groupby([col("groups")])
        .agg([col("a").mean()])
        .sort("a", Default::default())
        .collect()
        .unwrap();

    assert_eq!(out.column("a").unwrap().f64().unwrap().get(0), Some(1.0));
}

#[test]
fn test_lazy_tail() {
    let df = df! {
        "A" => &[1, 2, 3, 4, 5],
        "B" => &[5, 4, 3, 2, 1]
    }
    .unwrap();

    let _out = df.lazy().tail(3).collect().unwrap();
}

#[test]
fn test_lazy_groupby_sort() {
    let df = df! {
        "a" => ["a", "b", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }
    .unwrap();

    let out = df
        .clone()
        .lazy()
        .groupby([col("a")])
        .agg([col("b").sort(false).first()])
        .collect()
        .unwrap()
        .sort(["a"], false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b").unwrap().i32().unwrap()),
        [Some(1), Some(2), Some(6)]
    );

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b").sort(false).last()])
        .collect()
        .unwrap()
        .sort(["a"], false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b").unwrap().i32().unwrap()),
        [Some(3), Some(5), Some(6)]
    );
}

#[test]
fn test_lazy_groupby_sort_by() {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6],
        "c" => [6, 1, 4, 3, 2, 1]
    }
    .unwrap();

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b").sort_by([col("c")], [true]).first()])
        .collect()
        .unwrap()
        .sort(["a"], false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b").unwrap().i32().unwrap()),
        [Some(1), Some(4), Some(6)]
    );
}

#[test]
#[cfg(feature = "dtype-datetime")]
fn test_lazy_groupby_cast() {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }
    .unwrap();

    // test if it runs in groupby context
    let _out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b")
            .mean()
            .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_groupby_binary_expr() {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }
    .unwrap();

    // test if it runs in groupby context
    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b").mean() * lit(2)])
        .sort("a", Default::default())
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("b").unwrap().f64().unwrap()),
        [Some(4.0), Some(9.0), Some(12.0)]
    );
}

#[test]
fn test_lazy_groupby_filter() -> PolarsResult<()> {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }?;

    // We test if the filters work in the groupby context
    // and that the aggregations can deal with empty sets

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([
            col("b").filter(col("a").eq(lit("a"))).sum().alias("b_sum"),
            col("b")
                .filter(col("a").eq(lit("a")))
                .first()
                .alias("b_first"),
            col("b")
                .filter(col("a").eq(lit("e")))
                .mean()
                .alias("b_mean"),
            col("b")
                .filter(col("a").eq(lit("a")))
                .last()
                .alias("b_last"),
        ])
        .sort(
            "a",
            SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
            },
        )
        .collect()?;

    assert_eq!(
        Vec::from(out.column("b_sum").unwrap().i32().unwrap()),
        [Some(6), None, None]
    );
    assert_eq!(
        Vec::from(out.column("b_first").unwrap().i32().unwrap()),
        [Some(1), None, None]
    );
    assert_eq!(
        Vec::from(out.column("b_mean").unwrap().f64().unwrap()),
        [None, None, None]
    );
    assert_eq!(
        Vec::from(out.column("b_last").unwrap().i32().unwrap()),
        [Some(3), None, None]
    );

    Ok(())
}

#[test]
fn test_groupby_projection_pd_same_column() -> PolarsResult<()> {
    // this query failed when projection pushdown was enabled

    let a = || {
        let df = df![
            "col1" => ["a", "ab", "abc"],
            "col2" => [1, 2, 3]
        ]
        .unwrap();

        df.lazy()
            .select([col("col1").alias("foo"), col("col2").alias("bar")])
    };

    let out = a()
        .left_join(a(), col("foo"), col("foo"))
        .select([col("bar")])
        .collect()?;

    let a = out.column("bar")?.i32()?;
    assert_eq!(Vec::from(a), &[Some(1), Some(2), Some(3)]);

    Ok(())
}

#[test]
fn test_groupby_sort_slice() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 2, 2, 3, 3, 3],
        "vals" => [1, 5, 6, 3, 9, 8]
    ]?;
    // get largest two values per groups

    // expected:
    // group      values
    // 1          1
    // 2          6, 5
    // 3          9, 8

    let out1 = df
        .clone()
        .lazy()
        .sort(
            "vals",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .groupby([col("groups")])
        .agg([col("vals").head(Some(2)).alias("foo")])
        .sort("groups", SortOptions::default())
        .collect()?;

    let out2 = df
        .lazy()
        .groupby([col("groups")])
        .agg([col("vals").sort(true).head(Some(2)).alias("foo")])
        .sort("groups", SortOptions::default())
        .collect()?;

    assert!(out1.column("foo")?.series_equal(out2.column("foo")?));
    Ok(())
}

#[test]
fn test_groupby_cumsum() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 2, 2, 3, 3, 3],
        "vals" => [1, 5, 6, 3, 9, 8]
    ]?;

    let out = df
        .lazy()
        .groupby([col("groups")])
        .agg([col("vals").cumsum(false)])
        .sort("groups", Default::default())
        .collect()?;

    assert_eq!(
        Vec::from(out.column("vals")?.explode()?.i32()?),
        [1, 5, 11, 3, 12, 20]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );

    Ok(())
}

#[test]
fn test_arg_sort_multiple() -> PolarsResult<()> {
    let df = df![
        "int" => [1, 2, 3, 1, 2],
        "flt" => [3.0, 2.0, 1.0, 2.0, 1.0],
        "str" => ["a", "a", "a", "b", "b"]
    ]?;

    let out = df
        .clone()
        .lazy()
        .select([arg_sort_by([col("int"), col("flt")], &[true, false])])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("int")?.idx()?),
        [2, 4, 1, 3, 0]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );

    // check if this runs
    let _out = df
        .lazy()
        .select([arg_sort_by([col("str"), col("flt")], &[true, false])])
        .collect()?;
    Ok(())
}

#[test]
fn test_multiple_explode() -> PolarsResult<()> {
    let df = df![
        "a" => [0, 1, 2, 0, 2],
        "b" => [5, 4, 3, 2, 1],
        "c" => [2, 3, 4, 1, 5]
    ]?;

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b").alias("b_list"), col("c").alias("c_list")])
        .explode([col("c_list"), col("b_list")])
        .collect()?;
    assert_eq!(out.shape(), (5, 3));

    Ok(())
}

#[test]
fn test_filter_and_alias() -> PolarsResult<()> {
    let df = df![
        "a" => [0, 1, 2, 0, 2]
    ]?;

    let out = df
        .lazy()
        .with_column(col("a").pow(2.0).alias("a_squared"))
        .filter(col("a_squared").gt(lit(1)).and(col("a").gt(lit(1))))
        .collect()?;

    let expected = df![
        "a" => [2, 2],
        "a_squared" => [4.0, 4.0]
    ]?;
    println!("{:?}", out);
    println!("{:?}", expected);
    assert!(out.frame_equal(&expected));
    Ok(())
}

#[test]
fn test_filter_lit() {
    // see https://github.com/pola-rs/polars/issues/790
    // failed due to broadcasting filters and splitting threads.
    let iter = (0..100).map(|i| ('A'..='Z').nth(i % 26).unwrap().to_string());
    let a = Series::from_iter(iter);
    let df = DataFrame::new([a].into()).unwrap();

    let out = df.lazy().filter(lit(true)).collect().unwrap();
    assert_eq!(out.shape(), (100, 1));
}

#[test]
fn test_ternary_null() -> PolarsResult<()> {
    let df = df![
        "a" => ["a", "b", "c"]
    ]?;

    let out = df
        .lazy()
        .select([when(col("a").eq(lit("c")))
            .then(Null {}.lit())
            .otherwise(col("a"))
            .alias("foo")])
        .collect()?;

    assert_eq!(
        out.column("foo")?.is_null().into_iter().collect::<Vec<_>>(),
        &[Some(false), Some(false), Some(true)]
    );
    Ok(())
}

#[test]
fn test_fill_forward() -> PolarsResult<()> {
    let df = df![
        "a" => ["a", "b", "a"],
        "b" => [Some(1), None, None]
    ]?;

    let out = df
        .lazy()
        .select([col("b").forward_fill(None).list().over([col("a")])])
        .collect()?;
    let agg = out.column("b")?.list()?;

    let a: Series = agg.get(0).unwrap();
    assert!(a.series_equal(&Series::new("b", &[1, 1])));
    let a: Series = agg.get(2).unwrap();
    assert!(a.series_equal(&Series::new("b", &[1, 1])));
    let a: Series = agg.get(1).unwrap();
    assert_eq!(a.null_count(), 1);
    Ok(())
}

#[cfg(feature = "cross_join")]
#[test]
fn test_cross_join() -> PolarsResult<()> {
    let df1 = df![
        "a" => ["a", "b", "a"],
        "b" => [Some(1), None, None]
    ]?;

    let df2 = df![
        "a" => [1, 2],
        "b" => [None, Some(12)]
    ]?;

    let out = df1.lazy().cross_join(df2.lazy()).collect()?;
    assert_eq!(out.shape(), (6, 4));
    Ok(())
}

#[test]
fn test_select_empty_df() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/1056
    let df1 = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df1
        .lazy()
        .filter(col("a").eq(lit(0))) // this will lead to an empty frame
        .select([col("a"), lit(1).alias("c")])
        .collect()?;

    assert_eq!(out.column("a")?.len(), 0);
    assert_eq!(out.column("c")?.len(), 0);

    Ok(())
}

#[test]
fn test_keep_name() -> PolarsResult<()> {
    let df = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .select([
            col("a").alias("bar").keep_name(),
            col("b").alias("bar").keep_name(),
        ])
        .collect()?;

    assert_eq!(out.get_column_names(), &["a", "b"]);
    Ok(())
}

#[test]
fn test_exclude() -> PolarsResult<()> {
    let df = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3],
    "c" => [1, 2, 3]
    ]?;

    let out = df.lazy().select([col("*").exclude(&["b"])]).collect()?;

    assert_eq!(out.get_column_names(), &["a", "c"]);
    Ok(())
}

#[test]
#[cfg(feature = "regex")]
fn test_regex_selection() -> PolarsResult<()> {
    let df = df![
    "anton" => [1, 2, 3],
    "arnold schwars" => [1, 2, 3],
    "annie" => [1, 2, 3]
    ]?;

    let out = df.lazy().select([col("^a.*o.*$")]).collect()?;

    assert_eq!(out.get_column_names(), &["anton", "arnold schwars"]);
    Ok(())
}

#[test]
fn test_sort_by() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 1, 1, 2, 2],
        "c" => [2, 3, 1, 2, 1]
    ]?;

    // evaluate
    let out = df
        .clone()
        .lazy()
        .select([col("a").sort_by([col("b"), col("c")], [false])])
        .collect()?;

    let a = out.column("a")?;
    assert_eq!(
        Vec::from(a.i32().unwrap()),
        &[Some(3), Some(1), Some(2), Some(5), Some(4)]
    );

    // aggregate
    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("b")])
        .agg([col("a").sort_by([col("b"), col("c")], [false])])
        .collect()?;
    let a = out.column("a")?.explode()?;
    assert_eq!(
        Vec::from(a.i32().unwrap()),
        &[Some(3), Some(1), Some(2), Some(5), Some(4)]
    );

    // evaluate_on_groups
    let out = df
        .lazy()
        .groupby_stable([col("b")])
        .agg([col("a").sort_by([col("b"), col("c")], [false])])
        .collect()?;

    let a = out.column("a")?.explode()?;
    assert_eq!(
        Vec::from(a.i32().unwrap()),
        &[Some(3), Some(1), Some(2), Some(5), Some(4)]
    );

    Ok(())
}

#[test]
fn test_filter_after_shift_in_groups() -> PolarsResult<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .select([
            col("fruits"),
            col("B")
                .shift(1)
                .filter(col("B").shift(1).gt(lit(4)))
                .list()
                .over([col("fruits")])
                .alias("filtered"),
        ])
        .collect()?;

    assert_eq!(
        out.column("filtered")?
            .list()?
            .get(0)
            .unwrap()
            .i32()?
            .get(0)
            .unwrap(),
        5
    );
    assert_eq!(
        out.column("filtered")?
            .list()?
            .get(1)
            .unwrap()
            .i32()?
            .get(0)
            .unwrap(),
        5
    );
    assert_eq!(out.column("filtered")?.list()?.get(2).unwrap().len(), 0);

    Ok(())
}

#[test]
fn test_lazy_ternary_predicate_pushdown() -> PolarsResult<()> {
    let df = df![
        "a" => &[10, 1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .select([when(col("a").eq(lit(10)))
            .then(Null {}.lit())
            .otherwise(col("a"))])
        .drop_nulls(None)
        .collect()?;

    assert_eq!(
        Vec::from(out.get_columns()[0].i32()?),
        &[Some(1), Some(2), Some(3)]
    );

    Ok(())
}

#[test]
fn test_categorical_addition() -> PolarsResult<()> {
    let df = fruits_cars();

    // test if we can do that arithmetic operation with utf8 and categorical
    let out = df
        .lazy()
        .select([
            col("fruits").cast(DataType::Categorical(None)),
            col("cars").cast(DataType::Categorical(None)),
        ])
        .select([(col("fruits") + lit(" ") + col("cars")).alias("foo")])
        .collect()?;

    assert_eq!(out.column("foo")?.utf8()?.get(0).unwrap(), "banana beetle");

    Ok(())
}

#[test]
fn test_error_duplicate_names() {
    let df = fruits_cars();
    assert!(df.lazy().select([col("*"), col("*"),]).collect().is_err());
}

#[test]
fn test_filter_count() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("fruits")
            .filter(col("fruits").eq(lit("banana")))
            .count()])
        .collect()?;
    assert_eq!(out.column("fruits")?.u32()?.get(0), Some(3));
    Ok(())
}

#[test]
#[cfg(feature = "dtype-i16")]
fn test_groupby_small_ints() -> PolarsResult<()> {
    let df = df![
        "id_32" => [1i32, 2],
        "id_16" => [1i16, 2]
    ]?;

    // https://github.com/pola-rs/polars/issues/1255
    let out = df
        .lazy()
        .groupby([col("id_16"), col("id_32")])
        .agg([col("id_16").sum().alias("foo")])
        .sort(
            "foo",
            SortOptions {
                descending: true,
                ..Default::default()
            },
        )
        .collect()?;

    assert_eq!(Vec::from(out.column("foo")?.i64()?), &[Some(2), Some(1)]);
    Ok(())
}

#[test]
fn test_when_then_schema() -> PolarsResult<()> {
    let df = fruits_cars();

    let schema = df
        .lazy()
        .select([when(col("A").gt(lit(1)))
            .then(Null {}.lit())
            .otherwise(col("A"))])
        .schema();
    assert_ne!(schema?.get_index(0).unwrap().1, &DataType::Null);

    Ok(())
}

#[test]
fn test_singleton_broadcast() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("fruits"), lit(1).alias("foo")])
        .collect()?;

    assert!(out.column("foo")?.len() > 1);
    Ok(())
}

#[test]
fn test_sort_by_suffix() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("*")
            .sort_by([col("A")], [false])
            .list()
            .over([col("fruits")])
            .flatten()
            .suffix("_sorted")])
        .collect()?;

    let expected = df!(
            "A_sorted"=> [1, 2, 5, 3, 4],
            "fruits_sorted"=> ["banana", "banana", "banana", "apple", "apple"],
            "B_sorted"=> [5, 4, 1, 3, 2],
            "cars_sorted"=> ["beetle", "audi", "beetle", "beetle", "beetle"]
    )?;

    assert!(expected.frame_equal(&out));
    Ok(())
}

#[test]
fn test_list_in_select_context() -> PolarsResult<()> {
    let s = Series::new("a", &[1, 2, 3]);
    let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name()).unwrap();
    builder.append_series(&s);
    let expected = builder.finish().into_series();

    let df = DataFrame::new(vec![s])?;

    let out = df.lazy().select([col("a").list()]).collect()?;

    let s = out.column("a")?;
    assert!(s.series_equal(&expected));

    Ok(())
}

#[test]
fn test_round_after_agg() -> PolarsResult<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .groupby([col("fruits")])
        .agg([col("A")
            .cast(DataType::Float32)
            .mean()
            .round(2)
            .alias("foo")])
        .collect()?;

    assert!(out.column("foo")?.f32().is_ok());

    let df = df![
        "groups" => ["pigeon",
                 "rabbit",
                 "rabbit",
                 "Chris",
                 "pigeon",
                 "fast",
                 "fast",
                 "pigeon",
                 "rabbit",
                 "Chris"],
        "b" => [5409, 4848, 4864, 3540, 8103, 3083, 8575, 9963, 8809, 5425],
        "c" => [0.4517241160719615,
                  0.2551467646274673,
                  0.8682045191407308,
                  0.9925316385786037,
                  0.5392027792928116,
                  0.7633847828107002,
                  0.7967295231651537,
                  0.01444779067224733,
                  0.23807484087472652,
                  0.10985868798350984]
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("groups")])
        .agg([((col("b") * col("c")).sum() / col("b").sum())
            .round(2)
            .alias("foo")])
        .collect()?;

    let out = out.column("foo")?;
    let out = out.f64()?;

    assert_eq!(
        Vec::from(out),
        &[Some(0.3), Some(0.41), Some(0.46), Some(0.79)]
    );

    Ok(())
}

#[test]
#[cfg(feature = "dtype-date")]
fn test_fill_nan() -> PolarsResult<()> {
    let s0 = Series::new("date", &[1, 2, 3]).cast(&DataType::Date)?;
    let s1 = Series::new("float", &[Some(1.0), Some(f32::NAN), Some(3.0)]);

    let df = DataFrame::new(vec![s0, s1])?;
    let out = df.lazy().fill_nan(Null {}.lit()).collect()?;
    let out = out.column("float")?;
    assert_eq!(Vec::from(out.f32()?), &[Some(1.0), None, Some(3.0)]);

    Ok(())
}

#[test]
fn test_exclude_regex() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("*").exclude(["^(fruits|cars)$"])])
        .collect()?;

    assert_eq!(out.get_column_names(), &["A", "B"]);
    Ok(())
}

#[test]
fn test_groupby_rank() -> PolarsResult<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .groupby_stable([col("cars")])
        .agg([col("B").rank(RankOptions {
            method: RankMethod::Dense,
            ..Default::default()
        })])
        .collect()?;

    let out = out.column("B")?;
    let out = out.list()?.get(1).unwrap();
    let out = out.idx()?;

    assert_eq!(Vec::from(out), &[Some(1)]);
    Ok(())
}

#[test]
pub fn test_select_by_dtypes() -> PolarsResult<()> {
    let df = df![
        "bools" => [true, false, true],
        "ints" => [1, 2, 3],
        "strings" => ["a", "b", "c"],
        "floats" => [1.0, 2.0, 3.0f32]
    ]?;
    let out = df
        .lazy()
        .select([dtype_cols([DataType::Float32, DataType::Utf8])])
        .collect()?;
    assert_eq!(out.dtypes(), &[DataType::Utf8, DataType::Float32]);

    Ok(())
}

#[test]
fn test_binary_expr() -> PolarsResult<()> {
    // test panic in schema names
    let df = fruits_cars();
    let _ = df.lazy().select([col("A").neq(lit(1))]).collect()?;

    // test type coercion
    // https://github.com/pola-rs/polars/issues/1649
    let df = df!(
            "nrs"=> [Some(1i64), Some(2), Some(3), None, Some(5)],
            "random"=> [0.1f64, 0.6, 0.2, 0.6, 0.3]
    )?;

    let out = df
        .lazy()
        .select([when(col("random").gt(lit(0.5)))
            .then(lit(2))
            .otherwise(col("random"))
            .alias("other")
            * col("nrs").sum()])
        .collect()?;
    assert_eq!(out.dtypes(), &[DataType::Float64]);
    Ok(())
}

#[test]
fn test_single_group_result() -> PolarsResult<()> {
    // the arg_sort should not auto explode
    let df = df![
        "a" => [1, 2],
        "b" => [1, 1]
    ]?;

    let out = df
        .lazy()
        .select([col("a")
            .arg_sort(SortOptions {
                descending: false,
                nulls_last: false,
                multithreaded: true,
            })
            .list()
            .over([col("a")])
            .flatten()])
        .collect()?;

    let a = out.column("a")?.idx()?;
    assert_eq!(Vec::from(a), &[Some(0), Some(0)]);

    Ok(())
}

#[test]
fn test_single_ranked_group() -> PolarsResult<()> {
    // tests type consistency of rank algorithm
    let df = df!["group" => [1, 2, 2],
        "value"=> [100, 50, 10]
    ]?;

    let out = df
        .lazy()
        .with_columns([col("value")
            .rank(RankOptions {
                method: RankMethod::Average,
                ..Default::default()
            })
            .list()
            .over([col("group")])])
        .collect()?;

    let out = out.column("value")?.explode()?;
    let out = out.f32()?;
    assert_eq!(
        Vec::from(out),
        &[Some(1.0), Some(2.0), Some(1.0), Some(2.0), Some(1.0)]
    );

    Ok(())
}

#[test]
fn empty_df() -> PolarsResult<()> {
    let df = fruits_cars();
    let df = df.filter(&BooleanChunked::full("", false, df.height()))?;

    df.lazy()
        .select([
            col("A").shift(1).alias("1"),
            col("A").shift_and_fill(1, lit(1)).alias("2"),
            col("A").shift_and_fill(-1, lit(1)).alias("3"),
            col("A").fill_null(lit(1)).alias("4"),
            col("A").cumcount(false).alias("5"),
            col("A").diff(1, NullBehavior::Ignore).alias("6"),
            col("A").cummax(false).alias("7"),
            col("A").cummin(false).alias("8"),
        ])
        .collect()?;

    Ok(())
}

#[test]
fn test_apply_flatten() -> PolarsResult<()> {
    let df = df![
         "A"=> [1.1435, 2.223456, 3.44732, -1.5234, -2.1238, -3.2923],
        "B"=> ["a", "b", "a", "b", "a", "b"]
    ]?;

    let out = df
        .lazy()
        .groupby_stable([col("B")])
        .agg([col("A").abs().sum().alias("A_sum")])
        .collect()?;

    let out = out.column("A_sum")?;
    assert_eq!(out.get(0)?, AnyValue::Float64(6.71462));
    assert_eq!(out.get(1)?, AnyValue::Float64(7.039156));

    Ok(())
}

#[test]
#[cfg(feature = "is_in")]
fn test_is_in() -> PolarsResult<()> {
    let df = fruits_cars();

    // // this will be executed by apply
    let out = df
        .clone()
        .lazy()
        .groupby_stable([col("fruits")])
        .agg([col("cars").is_in(col("cars").filter(col("cars").eq(lit("beetle"))))])
        .collect()?;
    let out = out.column("cars").unwrap();
    let out = out.explode()?;
    let out = out.bool().unwrap();
    assert_eq!(
        Vec::from(out),
        &[Some(true), Some(false), Some(true), Some(true), Some(true)]
    );

    // this will be executed by map
    let out = df
        .lazy()
        .groupby_stable([col("fruits")])
        .agg([col("cars").is_in(lit(Series::new("a", ["beetle", "vw"])))])
        .collect()?;

    let out = out.column("cars").unwrap();
    let out = out.explode()?;
    let out = out.bool().unwrap();
    assert_eq!(
        Vec::from(out),
        &[Some(true), Some(false), Some(true), Some(true), Some(true)]
    );

    Ok(())
}

#[test]
fn test_partitioned_gb_1() -> PolarsResult<()> {
    // don't move these to integration tests
    // keep these dtypes
    let out = df![
        "keys" => [1, 1, 1, 1, 2],
        "vals" => ["a", "b", "c", "a", "a"]
    ]?
    .lazy()
    .groupby([col("keys")])
    .agg([
        (col("vals").eq(lit("a"))).sum().alias("eq_a"),
        (col("vals").eq(lit("b"))).sum().alias("eq_b"),
    ])
    .sort("keys", Default::default())
    .collect()?;

    assert!(out.frame_equal(&df![
        "keys" => [1, 2],
        "eq_a" => [2 as IdxSize, 1],
        "eq_b" => [1 as IdxSize, 0],
    ]?));

    Ok(())
}

#[test]
fn test_partitioned_gb_count() -> PolarsResult<()> {
    // don't move these to integration tests
    let out = df![
        "col" => (0..100).map(|_| Some(0)).collect::<Int32Chunked>().into_series(),
    ]?
    .lazy()
    .groupby([col("col")])
    .agg([
        // we make sure to alias with a different name
        count().alias("counted"),
        col("col").count().alias("count2"),
    ])
    .collect()?;

    assert!(out.frame_equal(&df![
        "col" => [0],
        "counted" => [100 as IdxSize],
        "count2" => [100 as IdxSize],
    ]?));

    Ok(())
}

#[test]
fn test_partitioned_gb_mean() -> PolarsResult<()> {
    // don't move these to integration tests
    let out = df![
        "key" => (0..100).map(|_| Some(0)).collect::<Int32Chunked>().into_series(),
    ]?
    .lazy()
    .with_columns([lit("a").alias("str"), lit(1).alias("int")])
    .groupby([col("key")])
    .agg([
        col("str").mean().alias("mean_str"),
        col("int").mean().alias("mean_int"),
    ])
    .collect()?;

    assert_eq!(out.shape(), (1, 3));
    let str_col = out.column("mean_str")?;
    assert_eq!(str_col.get(0)?, AnyValue::Null);
    let int_col = out.column("mean_int")?;
    assert_eq!(int_col.get(0)?, AnyValue::Float64(1.0));

    Ok(())
}

#[test]
fn test_partitioned_gb_binary() -> PolarsResult<()> {
    // don't move these to integration tests
    let df = df![
        "col" => (0..20).map(|_| Some(0)).collect::<Int32Chunked>().into_series(),
    ]?;

    let out = df
        .clone()
        .lazy()
        .groupby([col("col")])
        .agg([(col("col") + lit(10)).sum().alias("sum")])
        .collect()?;

    assert!(out.frame_equal(&df![
        "col" => [0],
        "sum" => [200],
    ]?));

    let out = df
        .lazy()
        .groupby([col("col")])
        .agg([(col("col").cast(DataType::Float32) + lit(10.0))
            .sum()
            .alias("sum")])
        .collect()?;

    assert!(out.frame_equal(&df![
        "col" => [0],
        "sum" => [200.0 as f32],
    ]?));

    Ok(())
}

#[test]
fn test_partitioned_gb_ternary() -> PolarsResult<()> {
    // don't move these to integration tests
    let df = df![
        "col" => (0..20).map(|_| Some(0)).collect::<Int32Chunked>().into_series(),
        "val" => (0..20).map(|i| Some(i)).collect::<Int32Chunked>().into_series(),
    ]?;

    let out = df
        .clone()
        .lazy()
        .groupby([col("col")])
        .agg([when(col("val").gt(lit(10)))
            .then(lit(1))
            .otherwise(lit(0))
            .sum()
            .alias("sum")])
        .collect()?;

    assert!(out.frame_equal(&df![
        "col" => [0],
        "sum" => [9],
    ]?));

    Ok(())
}
