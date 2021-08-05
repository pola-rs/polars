use crate::functions::{argsort_by, pearson_corr};
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::tests::get_df;
#[cfg(feature = "temporal")]
use polars_core::utils::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_core::{df, prelude::*};

use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::prelude::*;
use std::iter::FromIterator;

fn scan_foods_csv() -> LazyFrame {
    let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
    LazyCsvReader::new(path.to_string()).finish()
}

#[test]
fn test_lazy_ternary() {
    let df = get_df()
        .lazy()
        .with_column(
            when(col("sepal.length").lt(lit(5.0)))
                .then(lit(10))
                .otherwise(lit(1))
                .alias("new"),
        )
        .collect()
        .unwrap();
    assert_eq!(Some(43), df.column("new").unwrap().sum::<i32>());
}

#[test]
fn test_lazy_with_column() {
    let df = get_df()
        .lazy()
        .with_column(lit(10).alias("foo"))
        .collect()
        .unwrap();
    println!("{:?}", df);
    assert_eq!(df.width(), 6);
    assert!(df.column("foo").is_ok());

    let df = get_df()
        .lazy()
        .with_column(lit(10).alias("foo"))
        .select(&[col("foo"), col("sepal.width")])
        .collect()
        .unwrap();
    println!("{:?}", df);
}

#[test]
fn test_lazy_exec() {
    let df = get_df();
    let new = df
        .clone()
        .lazy()
        .select(&[col("sepal.width"), col("variety")])
        .sort("sepal.width", false)
        .collect();
    println!("{:?}", new);

    let new = df
        .lazy()
        .filter(not(col("sepal.width").lt(lit(3.5))))
        .collect()
        .unwrap();

    let check = new.column("sepal.width").unwrap().f64().unwrap().gt(3.4);
    assert!(check.all_true())
}

#[test]
fn test_lazy_alias() {
    let df = get_df();
    let new = df
        .lazy()
        .select(&[col("sepal.width").alias("petals"), col("sepal.width")])
        .collect()
        .unwrap();
    assert_eq!(new.get_column_names(), &["petals", "sepal.width"]);
}

#[test]
fn test_lazy_melt() {
    let df = get_df();
    let out = df
        .lazy()
        .melt(
            vec!["petal.width".to_string(), "petal.length".to_string()],
            vec!["sepal.length".to_string(), "sepal.width".to_string()],
        )
        .filter(col("variable").eq(lit("sepal.length")))
        .select(vec![col("variable"), col("petal.width"), col("value")])
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (7, 3));
    dbg!(out);
}

#[test]
fn test_lazy_drop_nulls() {
    let df = df! {
        "foo" => &[Some(1), None, Some(3)],
        "bar" => &[Some(1), Some(2), None]
    }
    .unwrap();

    let new = df.clone().lazy().drop_nulls(None).collect().unwrap();
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
        .select(&[col("sepal.width").map(|s| Ok(s * 200.0), None)])
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
        .groupby(vec![col("variety")])
        .agg(vec![col("sepal.width").min()])
        .collect()
        .unwrap();

    println!("{:?}", new);
    assert_eq!(new.shape(), (1, 2));
}

#[test]
fn test_lazy_pushdown_through_agg() {
    // An aggregation changes the schema names, check if the pushdown succeeds.
    let df = get_df();
    let new = df
        .lazy()
        .groupby(vec![col("variety")])
        .agg(vec![
            col("sepal.length").min(),
            col("petal.length").min().alias("foo"),
        ])
        .select(&[col("foo")])
        // second selection is to test if optimizer can handle that
        .select(&[col("foo").alias("bar")])
        .collect()
        .unwrap();

    println!("{:?}", new);
}

#[test]
#[cfg(feature = "temporal")]
fn test_lazy_agg() {
    let s0 = Date32Chunked::parse_from_str_slice(
        "date",
        &[
            "2020-08-21",
            "2020-08-21",
            "2020-08-22",
            "2020-08-23",
            "2020-08-22",
        ],
        "%Y-%m-%d",
    )
    .into_series();
    let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
    let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
    let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

    let lf = df
        .lazy()
        .groupby(vec![col("date")])
        .agg(vec![
            col("rain").min(),
            col("rain").sum(),
            col("rain").quantile(0.5).alias("median_rain"),
        ])
        .sort("date", false);

    println!("{:?}", lf.describe_plan());
    println!("{:?}", lf.describe_optimized_plan());
    let new = lf.collect().unwrap();
    println!("{:?}", new);
}

#[test]
fn test_lazy_shift() {
    let df = get_df();
    let new = df
        .lazy()
        .select(&[col("sepal.width").alias("foo").shift(2)])
        .collect()
        .unwrap();
    assert_eq!(new.column("foo").unwrap().f64().unwrap().get(0), None);
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
    dbg!(new);
}

#[test]
fn test_lazy_binary_ops() {
    let df = df!("a" => &[1, 2, 3, 4, 5, ]).unwrap();
    let new = df
        .lazy()
        .select(&[col("a").eq(lit(2)).alias("foo")])
        .collect()
        .unwrap();
    assert_eq!(new.column("foo").unwrap().sum::<i32>(), Some(1));
}

fn load_df() -> DataFrame {
    df!("a" => &[1, 2, 3, 4, 5],
                 "b" => &["a", "a", "b", "c", "c"],
                 "c" => &[1, 2, 3, 4, 5]
    )
    .unwrap()
}

#[test]
fn test_lazy_query_1() {
    // test on aggregation pushdown
    // and a filter that is not in the projection
    let df_a = load_df();
    let df_b = df_a.clone();
    df_a.lazy()
        .left_join(df_b.lazy(), col("b"), col("b"))
        .filter(col("a").lt(lit(2)))
        .groupby(vec![col("b")])
        .agg(vec![col("b").first(), col("c").first()])
        .select(&[col("b"), col("c_first")])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_query_2() {
    let df = load_df();
    let ldf = df
        .lazy()
        .with_column(col("a").map(|s| Ok(s * 2), None).alias("foo"))
        .filter(col("a").lt(lit(2)))
        .select(&[col("b"), col("a")]);

    let new = ldf.collect().unwrap();
    assert_eq!(new.shape(), (1, 2));
}

#[test]
fn test_lazy_query_3() {
    // query checks if schema of scanning is not changed by aggregation
    let _ = scan_foods_csv()
        .groupby(vec![col("calories")])
        .agg(vec![col("fats_g").max()])
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
        .groupby(vec![col("uid")])
        .agg(vec![
            col("day").list().alias("day"),
            col("cumcases")
                .map(
                    |s: Series| {
                        // determine the diff per column
                        let a: ListChunked = s
                            .list()
                            .unwrap()
                            .into_iter()
                            .map(|opt_s| opt_s.map(|s| &s - &(s.shift(1))))
                            .collect();
                        Ok(a.into_series())
                    },
                    None,
                )
                .alias("diff_cases"),
        ])
        .explode(vec![col("day"), col("diff_cases")])
        .join(
            base_df,
            vec![col("uid"), col("day")],
            vec![col("uid"), col("day")],
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
        .groupby(vec![col("uid")])
        .agg(vec![col("day").head(Some(2))])
        .collect()
        .unwrap();
    dbg!(&out);
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
fn test_lazy_query_6() -> Result<()> {
    let df = df! {
        "uid" => [0, 0, 0, 1, 1, 1],
        "day" => [1, 2, 4, 1, 2, 3],
        "cumcases" => [10, 12, 15, 25, 30, 41]
    }
    .unwrap();

    let out = df
        .lazy()
        .groupby(vec![col("uid")])
        // a double aggregation expression.
        .agg(vec![pearson_corr(col("day"), col("cumcases")).pow(2.0)])
        .sort("uid", false)
        .collect()
        .unwrap();
    let s = out.column("pearson_corr")?.f64()?;
    assert!((s.get(0).unwrap() - 0.994360902255639).abs() < 0.000001);
    assert!((s.get(1).unwrap() - 0.9552238805970149).abs() < 0.000001);
    Ok(())
}

#[test]
#[cfg(feature = "is_in")]
fn test_lazy_query_8() -> Result<()> {
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
fn test_lazy_query_9() -> Result<()> {
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
            vec![col("Sales.City")],
            vec![col("Cities.City")],
            JoinType::Inner,
        )
        .groupby(vec![col("Cities.Country")])
        .agg(vec![col("Sales.Amount").sum().alias("sum")])
        .sort("sum", false)
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
#[cfg(feature = "temporal")]
fn test_lazy_query_7() {
    let date = NaiveDate::from_ymd(2021, 3, 5);
    let dates = vec![
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 0, 0)),
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 1, 0)),
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 2, 0)),
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 3, 0)),
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 4, 0)),
        NaiveDateTime::new(date, NaiveTime::from_hms(12, 5, 0)),
    ];
    let data = vec![Some(1.), Some(2.), Some(3.), Some(4.), None, None];
    let df = DataFrame::new(vec![
        Date64Chunked::new_from_naive_datetime("date", &*dates).into(),
        Series::new("data", data),
    ])
    .unwrap();
    // this tests if predicate pushdown not interferes with the shift data.
    let out = df
        .lazy()
        .with_column(col("data").shift(-1).alias("output"))
        .with_column(col("output").shift(2).alias("shifted"))
        .filter(col("date").gt(lit(NaiveDateTime::new(date, NaiveTime::from_hms(12, 2, 0)))))
        .collect()
        .unwrap();
    let a = out.column(&"shifted").unwrap().sum::<f64>().unwrap() - 7.0;
    assert!(a < 0.01 && a > -0.01);
}

#[test]
fn test_lazy_shift_and_fill_all() {
    let data = &[1, 2, 3];
    let df = DataFrame::new(vec![Series::new("data", data)]).unwrap();
    let out = df
        .lazy()
        .with_column(col("data").shift(1).fill_none(lit(0)).alias("output"))
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("output").unwrap().i32().unwrap()),
        vec![Some(0), Some(2), Some(3)]
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
    let mut lp_top = to_alp(plan, &mut expr_arena, &mut lp_arena);
    lp_top = optimizer.optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top);
    let plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
    assert!(
        matches!(plan, LogicalPlan::Projection{ expr, ..} if matches!(&expr[0], Expr::BinaryExpr{left, ..} if **left == Expr::Literal(LiteralValue::Float32(2.0))))
    );
}

#[test]
fn test_lazy_wildcard() {
    let df = load_df();
    let new = df.clone().lazy().select(&[col("*")]).collect().unwrap();
    assert_eq!(new.shape(), (5, 3));

    let new = df
        .lazy()
        .groupby(vec![col("b")])
        .agg(vec![col("*").sum(), col("*").first()])
        .collect()
        .unwrap();
    assert_eq!(new.shape(), (3, 6));
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
fn test_lazy_filter_and_rename() {
    let df = load_df();
    let lf = df
        .clone()
        .lazy()
        .with_column_renamed("a", "x")
        .filter(col("x").map(
            |s: Series| Ok(s.gt(3).into_series()),
            Some(DataType::Boolean),
        ))
        .select(&[col("x")]);

    let correct = df! {
        "x" => &[4, 5]
    }
    .unwrap();
    assert!(lf.collect().unwrap().frame_equal(&correct));

    // now we check if the column is rename or added when we don't select
    let lf = df.lazy().with_column_renamed("a", "x").filter(col("x").map(
        |s: Series| Ok(s.gt(3).into_series()),
        Some(DataType::Boolean),
    ));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
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
        .quantile(0.5)
        .collect()
        .unwrap()
        .frame_equal_missing(&df.quantile(0.5).unwrap()));
}

#[test]
fn test_lazy_predicate_pushdown_binary_expr() {
    let df = load_df();
    df.lazy()
        .filter(col("a").eq(col("b")))
        .select(&[col("c")])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_update_column() {
    let df = load_df();
    df.lazy().with_column(col("a") / lit(10)).collect().unwrap();
}

#[test]
fn test_lazy_fill_none() {
    let df = df! {
        "a" => &[None, Some(2)],
        "b" => &[Some(1), None]
    }
    .unwrap();
    let out = df.lazy().fill_none(lit(10.0)).collect().unwrap();
    let correct = df! {
        "a" => &[Some(10.0), Some(2.0)],
        "b" => &[Some(1.0), Some(10.0)]
    }
    .unwrap();
    assert!(out.frame_equal(&correct));
    assert_eq!(out.get_column_names(), vec!["a", "b"])
}

#[test]
fn test_lazy_window_functions() {
    let df = df! {
        "groups" => &[1, 1, 2, 2, 1, 2, 3, 3, 1],
        "values" => &[1, 2, 3, 4, 5, 6, 7, 8, 8]
    }
    .unwrap();

    // sums
    // 1 => 16
    // 2 => 13
    // 3 => 15
    let correct = [16, 16, 13, 13, 16, 13, 15, 15, 16]
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();

    // test if groups is available after projection pushdown.
    let _ = df
        .clone()
        .lazy()
        .select(&[avg("values").over(vec![col("groups")]).alias("part")])
        .collect()
        .unwrap();
    // test if partition aggregation is correct
    let out = df
        .lazy()
        .select(&[col("groups"), sum("values").over(vec![col("groups")])])
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.select_at_idx(1).unwrap().i32().unwrap()),
        correct
    );
    dbg!(out);
}

#[test]
fn test_lazy_double_projection() {
    let df = df! {
        "foo" => &[1, 2, 3]
    }
    .unwrap();
    df.lazy()
        .select(&[col("foo").alias("bar")])
        .select(&[col("bar")])
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

    let lp = df.lazy().select(&[col("foo") * col("bar")]).logical_plan;

    let mut expr_arena = Arena::new();
    let mut lp_arena = Arena::new();
    let rules: &mut [Box<dyn OptimizationRule>] = &mut [Box::new(TypeCoercionRule {})];

    let optimizer = StackOptimizer {};
    let mut lp_top = to_alp(lp, &mut expr_arena, &mut lp_arena);
    lp_top = optimizer.optimize_loop(rules, &mut expr_arena, &mut lp_arena, lp_top);
    let lp = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);

    if let LogicalPlan::Projection { expr, .. } = lp {
        if let Expr::BinaryExpr { left, right, .. } = &expr[0] {
            assert!(matches!(&**left, Expr::Cast { .. }));
            assert!(matches!(&**right, Expr::Cast { .. }));
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
        .groupby(vec![col("foo")])
        .agg(vec![col("bar").mean()])
        .sort("foo", false)
        .collect()
        .unwrap();

    assert_eq!(
        Vec::from(out.column("bar_mean").unwrap().f64().unwrap()),
        &[Some(1.0), Some(2.0), Some(3.0)]
    );

    let out = scan_foods_csv()
        .groupby(vec![col("category")])
        .agg(vec![col("calories").list()])
        .sort("category", false)
        .collect()
        .unwrap();
    dbg!(&out);
    let cat_agg_list = out.select_at_idx(1).unwrap();
    let fruit_series = cat_agg_list.list().unwrap().get(0).unwrap();
    let fruit_list = fruit_series.i64().unwrap();
    dbg!(fruit_list);
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
    let df = df! {
        "A" => &[1, 2, 3, 4, 5],
        "fruits" => &["banana", "banana", "apple", "apple", "banana"],
        "B" => &[5, 4, 3, 2, 1],
        "cars" => &["beetle", "audi", "beetle", "beetle", "beetle"]
    }
    .unwrap();

    df.lazy()
        .groupby(vec![col("fruits")])
        .agg(vec![col("cars").map(
            |s: Series| {
                let ca: UInt32Chunked = s
                    .list()?
                    .into_iter()
                    .map(|opt_s| opt_s.map(|s| s.len() as u32))
                    .collect();
                Ok(ca.into_series())
            },
            None,
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
        .clone()
        .lazy()
        .shift_and_fill(-1, col("B").std())
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
        .groupby(vec![col("groups")])
        .agg(vec![col("a").mean()])
        .sort("a_mean", false)
        .collect()
        .unwrap();

    assert_eq!(
        out.column("a_mean").unwrap().f64().unwrap().get(0),
        Some(1.0)
    );
}

#[test]
fn test_lazy_tail() {
    let df = df! {
        "A" => &[1, 2, 3, 4, 5],
        "B" => &[5, 4, 3, 2, 1]
    }
    .unwrap();

    let _out = df.clone().lazy().tail(3).collect().unwrap();
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
        .groupby(vec![col("a")])
        .agg(vec![col("b").sort(false).first()])
        .collect()
        .unwrap()
        .sort("a", false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b_first").unwrap().i32().unwrap()),
        [Some(1), Some(2), Some(6)]
    );

    let out = df
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![col("b").sort(false).last()])
        .collect()
        .unwrap()
        .sort("a", false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b_last").unwrap().i32().unwrap()),
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
        .groupby(vec![col("a")])
        .agg(vec![col("b").sort_by(col("c"), true).first()])
        .collect()
        .unwrap()
        .sort("a", false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b_first").unwrap().i32().unwrap()),
        [Some(1), Some(4), Some(6)]
    );
}

#[test]
#[cfg(feature = "dtype-date64")]
fn test_lazy_groupby_cast() {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }
    .unwrap();

    // test if it runs in groupby context
    let _out = df
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![col("b").mean().cast(DataType::Date64)])
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
        .groupby(vec![col("a")])
        .agg(vec![col("b").mean() * lit(2)])
        .sort("a", false)
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.column("b_mean").unwrap().f64().unwrap()),
        [Some(4.0), Some(9.0), Some(12.0)]
    );
}

#[test]
fn test_lazy_groupby_filter() -> Result<()> {
    let df = df! {
        "a" => ["a", "a", "a", "b", "b", "c"],
        "b" => [1, 2, 3, 4, 5, 6]
    }?;

    // We test if the filters work in the groupby context
    // and that the aggregations can deal with empty sets

    let out = df
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![
            col("b").filter(col("a").eq(lit("a"))).sum(),
            col("b").filter(col("a").eq(lit("a"))).first(),
            col("b").filter(col("a").eq(lit("e"))).mean(),
            col("b").filter(col("a").eq(lit("a"))).last(),
        ])
        .sort("a", false)
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
fn test_groupby_projection_pd_same_column() -> Result<()> {
    // this query failed when projection pushdown was enabled

    let a = || {
        let df = df![
            "col1" => ["a", "ab", "abc"],
            "col2" => [1, 2, 3]
        ]
        .unwrap();

        df.lazy()
            .select(vec![col("col1").alias("foo"), col("col2").alias("bar")])
    };

    let out = a()
        .left_join(a(), col("foo"), col("foo"))
        .select(vec![col("bar")])
        .collect()?;

    let a = out.column("bar")?.i32()?;
    assert_eq!(Vec::from(a), &[Some(1), Some(2), Some(3)]);

    Ok(())
}

#[test]
fn test_groupby_sort_slice() -> Result<()> {
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
        .sort("vals", true)
        .groupby(vec![col("groups")])
        .agg(vec![col("vals").head(Some(2)).alias("foo")])
        .sort("groups", false)
        .collect()?;

    let out2 = df
        .lazy()
        .groupby(vec![col("groups")])
        .agg(vec![col("vals").sort(true).head(Some(2)).alias("foo")])
        .sort("groups", false)
        .collect()?;

    assert!(out1.column("foo")?.series_equal(out2.column("foo")?));
    dbg!(out1, out2);
    Ok(())
}

#[test]
fn test_groupby_cumsum() -> Result<()> {
    let df = df![
        "groups" => [1, 2, 2, 3, 3, 3],
        "vals" => [1, 5, 6, 3, 9, 8]
    ]?;

    let out = df
        .lazy()
        .groupby(vec![col("groups")])
        .agg(vec![col("vals").cum_sum(false)])
        .sort("groups", false)
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
fn test_argsort_multiple() -> Result<()> {
    let df = df![
        "int" => [1, 2, 3, 1, 2],
        "flt" => [3.0, 2.0, 1.0, 2.0, 1.0],
        "str" => ["a", "a", "a", "b", "b"]
    ]?;

    let out = df
        .clone()
        .lazy()
        .select(vec![argsort_by(
            vec![col("int"), col("flt")],
            &[true, false],
        )])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("int")?.u32()?),
        [2, 4, 1, 3, 0]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );

    // check if this runs
    let _out = df
        .lazy()
        .select(vec![argsort_by(
            vec![col("str"), col("flt")],
            &[true, false],
        )])
        .collect()?;
    Ok(())
}

#[test]
fn test_multiple_explode() -> Result<()> {
    let df = df![
        "a" => [0, 1, 2, 0, 2],
        "b" => [5, 4, 3, 2, 1],
        "c" => [2, 3, 4, 1, 5]
    ]?;

    let out = df
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![
            col("b").list().alias("b_list"),
            col("c").list().alias("c_list"),
        ])
        .explode(vec![col("c_list"), col("b_list")])
        .collect()?;
    assert_eq!(out.shape(), (5, 3));

    Ok(())
}

#[test]
fn test_filter_and_alias() -> Result<()> {
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
        "a_squared" => [4, 4]
    ]?;

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
fn test_ternary_null() -> Result<()> {
    let df = df![
        "a" => ["a", "b", "c"]
    ]?;

    let out = df
        .lazy()
        .select(vec![when(col("a").eq(lit("c")))
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
fn test_fill_forward() -> Result<()> {
    let df = df![
        "a" => ["a", "b", "a"],
        "b" => [Some(1), None, None]
    ]?;

    let out = df
        .lazy()
        .select(vec![col("b").forward_fill().over(vec![col("a")])])
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
fn test_cross_join() -> Result<()> {
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
fn test_fold_wildcard() -> Result<()> {
    let df1 = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df1
        .clone()
        .lazy()
        .select(vec![
            fold_exprs(lit(0), |a, b| Ok(&a + &b), vec![col("*")]).alias("foo")
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("foo")?.i32()?),
        &[Some(2), Some(4), Some(6)]
    );

    // test if we don't panic due to wildcard
    let out = df1
        .lazy()
        .select(vec![all_exprs(vec![col("*").is_not_null()])])
        .collect()?;
    Ok(())
}

#[test]
fn test_select_empty_df() -> Result<()> {
    // https://github.com/pola-rs/polars/issues/1056
    let df1 = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df1
        .lazy()
        .filter(col("a").eq(lit(0))) // this will lead to an empty frame
        .select(vec![col("a"), lit(1).alias("c")])
        .collect()?;

    assert_eq!(out.column("a")?.len(), 0);
    assert_eq!(out.column("c")?.len(), 0);

    Ok(())
}

#[test]
fn test_keep_name() -> Result<()> {
    let df = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3]
    ]?;

    let out = df
        .lazy()
        .select(vec![
            col("a").alias("bar").keep_name(),
            col("b").alias("bar").keep_name(),
        ])
        .collect()?;

    assert_eq!(out.get_column_names(), &["a", "b"]);
    Ok(())
}

#[test]
fn test_exclude() -> Result<()> {
    let df = df![
    "a" => [1, 2, 3],
    "b" => [1, 2, 3],
    "c" => [1, 2, 3]
    ]?;

    let out = df.lazy().select(vec![col("*").exclude(&["b"])]).collect()?;

    assert_eq!(out.get_column_names(), &["a", "c"]);
    Ok(())
}

#[test]
#[cfg(feature = "regex")]
fn test_regex_selection() -> Result<()> {
    let df = df![
    "anton" => [1, 2, 3],
    "arnold schwars" => [1, 2, 3],
    "annie" => [1, 2, 3]
    ]?;

    let out = df.lazy().select(vec![col("^a.*o.*$")]).collect()?;

    assert_eq!(out.get_column_names(), &["anton", "arnold schwars"]);
    Ok(())
}

#[test]
fn test_filter_in_groupby_agg() -> Result<()> {
    // This tests if the fitler is correctly handled by the binary expression.
    // This could lead to UB if it were not the case. The filter creates an empty column.
    // but the group tuples could still be untouched leading to out of bounds aggregation.
    let df = df![
        "a" => [1, 1, 2],
        "b" => [1, 2, 3]
    ]?;

    let out = df
        .clone()
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![
            (col("b").filter(col("b").eq(lit(100))) * lit(2)).mean()
        ])
        .collect()?;

    assert_eq!(out.column("b_mean")?.null_count(), 2);

    let out = df
        .lazy()
        .groupby(vec![col("a")])
        .agg(vec![(col("b")
            .filter(col("b").eq(lit(100)))
            .map(|s| Ok(s), None))
        .mean()])
        .collect()?;
    assert_eq!(out.column("b_mean")?.null_count(), 2);

    Ok(())
}
