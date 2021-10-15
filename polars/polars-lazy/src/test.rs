use crate::functions::{argsort_by, pearson_corr};
use crate::logical_plan::optimizer::stack_opt::{OptimizationRule, StackOptimizer};
use crate::tests::get_df;
#[cfg(feature = "temporal")]
use polars_core::utils::chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_core::{df, prelude::*};

use crate::logical_plan::optimizer::simplify_expr::SimplifyExprRule;
use crate::prelude::*;
use polars_core::chunked_array::builder::get_list_builder;
use std::iter::FromIterator;

fn scan_foods_csv() -> LazyFrame {
    let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
    LazyCsvReader::new(path.to_string()).finish()
}

pub(crate) fn fruits_cars() -> DataFrame {
    df!(
            "A"=> [1, 2, 3, 4, 5],
            "fruits"=> ["banana", "banana", "apple", "apple", "banana"],
            "B"=> [5, 4, 3, 2, 1],
            "cars"=> ["beetle", "audi", "beetle", "beetle", "beetle"]
    )
    .unwrap()
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
        .select([col("variable"), col("petal.width"), col("value")])
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
        .select(&[col("sepal.width").map(|s| Ok(s * 200.0), GetOutput::same_type())])
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

    println!("{:?}", new);
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
    let s0 = DateChunked::parse_from_str_slice(
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
        .groupby([col("date")])
        .agg([
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
        .groupby([col("b")])
        .agg([col("b").first(), col("c").first()])
        .select(&[col("b"), col("c_first")])
        .collect()
        .unwrap();
}

#[test]
fn test_lazy_query_2() {
    let df = load_df();
    let ldf = df
        .lazy()
        .with_column(
            col("a")
                .map(|s| Ok(s * 2), GetOutput::same_type())
                .alias("foo"),
        )
        .filter(col("a").lt(lit(2)))
        .select(&[col("b"), col("a")]);

    let new = ldf.collect().unwrap();
    assert_eq!(new.shape(), (1, 2));
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
            col("day").list().alias("day"),
            col("cumcases")
                .apply(|s: Series| Ok(&s - &(s.shift(1))), GetOutput::same_type())
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
        .groupby([col("uid")])
        .agg([col("day").head(Some(2))])
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
        .groupby([col("uid")])
        // a double aggregation expression.
        .agg([pearson_corr(col("day"), col("cumcases")).pow(2.0)])
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
        .groupby([col("Cities.Country")])
        .agg([col("Sales.Amount").sum().alias("sum")])
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
        DatetimeChunked::new_from_naive_datetime("date", &*dates).into(),
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
        .groupby([col("b")])
        .agg([col("*").sum(), col("*").first()])
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
            GetOutput::from_type(DataType::Boolean),
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
        GetOutput::from_type(DataType::Boolean),
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
fn test_lazy_fill_null() {
    let df = df! {
        "a" => &[None, Some(2)],
        "b" => &[Some(1), None]
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
        .select(&[avg("values").over([col("groups")]).alias("part")])
        .collect()
        .unwrap();
    // test if partition aggregation is correct
    let out = df
        .lazy()
        .select(&[col("groups"), sum("values").over([col("groups")])])
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
        .sort("foo", false)
        .collect()
        .unwrap();

    assert_eq!(
        Vec::from(out.column("bar_mean").unwrap().f64().unwrap()),
        &[Some(1.0), Some(2.0), Some(3.0)]
    );

    let out = scan_foods_csv()
        .groupby([col("category")])
        .agg([col("calories").list()])
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
    let df = fruits_cars();

    df.lazy()
        .groupby([col("fruits")])
        .agg([col("cars").apply(
            |s: Series| Ok(Series::new("", &[s.len() as u32])),
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
        .groupby([col("groups")])
        .agg([col("a").mean()])
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
        .sort("a", false)
        .unwrap();

    assert_eq!(
        Vec::from(out.column("b_first").unwrap().i32().unwrap()),
        [Some(1), Some(2), Some(6)]
    );

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([col("b").sort(false).last()])
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
        .groupby([col("a")])
        .agg([col("b").sort_by(col("c"), true).first()])
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
        .agg([col("b").mean().cast(DataType::Datetime)])
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
        .groupby([col("a")])
        .agg([
            col("b").filter(col("a").eq(lit("a"))).sum(),
            col("b").filter(col("a").eq(lit("a"))).first(),
            col("b").filter(col("a").eq(lit("e"))).mean(),
            col("b").filter(col("a").eq(lit("a"))).last(),
        ])
        .sort("a", false)
        .collect()?;

    dbg!(&out);
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
        .groupby([col("groups")])
        .agg([col("vals").head(Some(2)).alias("foo")])
        .sort("groups", false)
        .collect()?;

    let out2 = df
        .lazy()
        .groupby([col("groups")])
        .agg([col("vals").sort(true).head(Some(2)).alias("foo")])
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
        .groupby([col("groups")])
        .agg([col("vals").cumsum(false)])
        .sort("groups", false)
        .collect()?;

    dbg!(&out);

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
        .select([argsort_by([col("int"), col("flt")], &[true, false])])
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
        .select([argsort_by([col("str"), col("flt")], &[true, false])])
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
        .groupby([col("a")])
        .agg([
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
fn test_fill_forward() -> Result<()> {
    let df = df![
        "a" => ["a", "b", "a"],
        "b" => [Some(1), None, None]
    ]?;

    let out = df
        .lazy()
        .select([col("b").forward_fill().over([col("a")])])
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
        .select([fold_exprs(lit(0), |a, b| Ok(&a + &b), vec![col("*")]).alias("foo")])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("foo")?.i32()?),
        &[Some(2), Some(4), Some(6)]
    );

    // test if we don't panic due to wildcard
    let _out = df1
        .lazy()
        .select([all_exprs(vec![col("*").is_not_null()])])
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
        .select([col("a"), lit(1).alias("c")])
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
        .select([
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

    let out = df.lazy().select([col("*").exclude(&["b"])]).collect()?;

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

    let out = df.lazy().select([col("^a.*o.*$")]).collect()?;

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
        .groupby([col("a")])
        .agg([(col("b").filter(col("b").eq(lit(100))) * lit(2)).mean()])
        .collect()?;

    assert_eq!(out.column("b_mean")?.null_count(), 2);

    let out = df
        .lazy()
        .groupby([col("a")])
        .agg([(col("b")
            .filter(col("b").eq(lit(100)))
            .map(Ok, GetOutput::same_type()))
        .mean()])
        .collect()?;
    assert_eq!(out.column("b_mean")?.null_count(), 2);

    Ok(())
}

#[test]
fn test_shift_and_fill_window_function() -> Result<()> {
    let df = fruits_cars();

    // a ternary expression with a final list aggregation
    let out1 = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("B").shift_and_fill(-1, lit(-1)).over([col("fruits")]),
        ])
        .collect()?;

    // same expression, no final list aggregation
    let out2 = df
        .lazy()
        .select([
            col("fruits"),
            col("B")
                .shift_and_fill(-1, lit(-1))
                .list()
                .over([col("fruits")]),
        ])
        .collect()?;

    dbg!(&out1, &out2);

    assert!(out1.frame_equal(&out2));

    Ok(())
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
fn test_exploded_window_function() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .clone()
        .lazy()
        .sort("fruits", false)
        .select([
            col("fruits"),
            col("B")
                .shift(1)
                .over([col("fruits")])
                .explode()
                .alias("shifted"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("shifted")?.i32()?),
        &[None, Some(3), None, Some(5), Some(4)]
    );

    // this tests if cast succeeds in aggregation context
    // we implicitly also test that a literal does not upcast a column
    let out = df
        .lazy()
        .sort("fruits", false)
        .select([
            col("fruits"),
            col("B")
                .shift_and_fill(1, lit(-1.0))
                .over([col("fruits")])
                .explode()
                .alias("shifted"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("shifted")?.i32()?),
        &[Some(-1), Some(3), Some(-1), Some(5), Some(4)]
    );
    Ok(())
}

#[test]
fn test_reverse_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("fruits", false)
        .select([col("B")
            .reverse()
            .over([col("fruits")])
            .explode()
            .alias("rev")])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("rev")?.i32()?),
        &[Some(2), Some(3), Some(1), Some(4), Some(5)]
    );
    Ok(())
}
#[test]
fn test_take_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("fruits", false)
        .select([col("B")
            .take(lit(Series::new("", &[0u32])))
            .over([col("fruits")])
            .explode()
            .alias("taken")])
        .collect()?;

    assert_eq!(Vec::from(out.column("taken")?.i32()?), &[Some(3), Some(5)]);
    Ok(())
}

#[test]
fn test_sort_by_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("cars", false)
        .select([
            col("fruits"),
            col("cars"),
            col("A")
                .sort_by(col("B"), false)
                .over([col("cars")])
                .explode()
                .alias("sorted_A_by_B"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("sorted_A_by_B")?.i32()?),
        &[Some(2), Some(5), Some(4), Some(3), Some(1)]
    );
    Ok(())
}

#[test]
fn test_filter_after_shift_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .select([
            col("fruits"),
            col("B")
                .shift(1)
                .filter(col("B").shift(1).gt(lit(4)))
                .over([col("fruits")])
                .alias("filtered"),
        ])
        .collect()?;
    dbg!(out.column("filtered")?);

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
fn test_lazy_ternary_predicate_pushdown() -> Result<()> {
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
fn test_categorical_addition() -> Result<()> {
    let df = fruits_cars();

    // test if we can do that arithmetic operation with utf8 and categorical
    let out = df
        .lazy()
        .select([
            col("fruits").cast(DataType::Categorical),
            col("cars").cast(DataType::Categorical),
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
fn test_filter_count() -> Result<()> {
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
fn test_groupby_small_ints() -> Result<()> {
    let df = df![
        "id_32" => [1i32, 2],
        "id_16" => [1i16, 2]
    ]?;

    // https://github.com/pola-rs/polars/issues/1255
    let out = df
        .lazy()
        .groupby([col("id_16"), col("id_32")])
        .agg([col("id_16").sum().alias("foo")])
        .sort("foo", true)
        .collect()?;

    assert_eq!(Vec::from(out.column("foo")?.i16()?), &[Some(2), Some(1)]);
    Ok(())
}

#[test]
fn test_when_then_schema() -> Result<()> {
    let df = fruits_cars();

    let schema = df
        .lazy()
        .select([when(col("A").gt(lit(1)))
            .then(Null {}.lit())
            .otherwise(col("A"))])
        .schema();
    assert_ne!(schema.fields()[0].data_type(), &DataType::Null);

    Ok(())
}

#[test]
fn test_singleton_broadcast() -> Result<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("fruits"), lit(1).alias("foo")])
        .collect()?;

    assert!(out.column("foo")?.len() > 1);
    Ok(())
}

#[test]
fn test_sort_by_suffix() -> Result<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("*")
            .sort_by(col("A"), false)
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
fn test_list_in_select_context() -> Result<()> {
    let s = Series::new("a", &[1, 2, 3]);
    let mut builder = get_list_builder(s.dtype(), s.len(), 1, s.name());
    builder.append_series(&s);
    let expected = builder.finish().into_series();

    let df = DataFrame::new(vec![s])?;

    let out = df.lazy().select([col("a").list()]).collect()?;

    let s = out.column("a")?;
    assert!(s.series_equal(&expected));

    Ok(())
}

#[test]
fn test_round_after_agg() -> Result<()> {
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
    Ok(())
}

#[test]
fn test_power_in_agg_list() -> Result<()> {
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
#[cfg(feature = "dtype-date")]
fn test_fill_nan() -> Result<()> {
    let s0 = Series::new("date", &[1, 2, 3]).cast(&DataType::Date)?;
    let s1 = Series::new("float", &[Some(1.0), Some(f32::NAN), Some(3.0)]);

    let df = DataFrame::new(vec![s0, s1])?;
    let out = df.lazy().fill_nan(Null {}.lit()).collect()?;
    let out = out.column("float")?;
    assert_eq!(Vec::from(out.f32()?), &[Some(1.0), None, Some(3.0)]);

    Ok(())
}

#[test]
fn test_agg_exprs() -> Result<()> {
    let df = fruits_cars();

    // a binary expression followed by a function and an aggregation. See if it runs
    let out = df
        .lazy()
        .stable_groupby([col("cars")])
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
fn test_exclude_regex() -> Result<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .select([col("*").exclude("^(fruits|cars)$")])
        .collect()?;

    assert_eq!(out.get_column_names(), &["A", "B"]);
    Ok(())
}

#[test]
fn test_groupby_rank() -> Result<()> {
    let df = fruits_cars();
    let out = df
        .lazy()
        .stable_groupby([col("cars")])
        .agg([col("B").rank(RankMethod::Dense)])
        .collect()?;

    let out = out.column("B")?;
    let out = out.list()?.get(1).unwrap();
    let out = out.u32()?;

    assert_eq!(Vec::from(out), &[Some(1)]);
    Ok(())
}

#[test]
fn test_apply_multiple_columns() -> Result<()> {
    let df = fruits_cars();

    let multiply = |s: &mut [Series]| Ok(&s[0].pow(2.0).unwrap() * &s[1]);

    let out = df
        .clone()
        .lazy()
        .select([map_mul(
            multiply,
            [col("A"), col("B")],
            GetOutput::from_type(DataType::Float64),
        )])
        .collect()?;
    let out = out.column("A")?;
    let out = out.f64()?;
    assert_eq!(
        Vec::from(out),
        &[Some(5.0), Some(16.0), Some(27.0), Some(32.0), Some(25.0)]
    );

    let out = df
        .lazy()
        .stable_groupby([col("cars")])
        .agg([apply_mul(
            multiply,
            [col("A"), col("B")],
            GetOutput::from_type(DataType::Float64),
        )])
        .collect()?;

    let out = out.column("A")?;
    let out = out.list()?.get(1).unwrap();
    let out = out.f64()?;

    assert_eq!(Vec::from(out), &[Some(16.0)]);
    Ok(())
}
