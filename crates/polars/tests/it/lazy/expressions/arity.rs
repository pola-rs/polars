use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

#[test]
#[cfg(feature = "unique_counts")]
fn test_list_broadcast() {
    // simply test if this runs
    df![
        "g" => [1, 1, 1],
        "a" => [1, 2, 3],
    ]
    .unwrap()
    .lazy()
    .group_by([col("g")])
    .agg([col("a").unique_counts() * len()])
    .collect()
    .unwrap();
}

#[test]
fn ternary_expand_sizes() -> PolarsResult<()> {
    let df = df! {
        "a" => [Some("a1"), None, None],
        "b" => [Some("b1"), Some("b2"), None]
    }?;
    let out = df
        .lazy()
        .with_column(
            when(not(lit(true)))
                .then(lit("unexpected"))
                .when(not(col("a").is_null()))
                .then(col("a"))
                .when(not(col("b").is_null()))
                .then(col("b"))
                .otherwise(lit("otherwise"))
                .alias("c"),
        )
        .collect()?;
    let vals = out.column("c")?.str()?.no_null_iter().collect::<Vec<_>>();
    assert_eq!(vals, &["a1", "b2", "otherwise"]);
    Ok(())
}

#[test]
fn ternary_short_circuit_is_opt_in() -> PolarsResult<()> {
    fn counted_branch(counter: Arc<AtomicUsize>, multiplier: i32) -> Expr {
        col("a").map(
            move |s| {
                counter.fetch_add(s.len(), Ordering::Relaxed);
                Ok(s.i32()?.apply_values(|v| v * multiplier).into_column())
            },
            |_, f| Ok(f.clone()),
        )
    }

    let df = df!["a" => [1i32, 2, 3, 4, 5]]?;

    let eager_first = Arc::new(AtomicUsize::new(0));
    let eager_second = Arc::new(AtomicUsize::new(0));
    let eager_fallback = Arc::new(AtomicUsize::new(0));

    let eager = df
        .clone()
        .lazy()
        .select([when(col("a").lt(lit(3)))
            .then(counted_branch(eager_first.clone(), 10))
            .when(col("a").lt(lit(5)))
            .then(counted_branch(eager_second.clone(), 100))
            .otherwise(counted_branch(eager_fallback.clone(), 1000))
            .alias("out")])
        .collect()?;

    assert_eq!(
        eager
            .column("out")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![10, 20, 300, 400, 5000]
    );
    assert_eq!(eager_first.load(Ordering::Relaxed), 5);
    assert_eq!(eager_second.load(Ordering::Relaxed), 5);
    assert_eq!(eager_fallback.load(Ordering::Relaxed), 5);

    let short_first = Arc::new(AtomicUsize::new(0));
    let short_second = Arc::new(AtomicUsize::new(0));
    let short_fallback = Arc::new(AtomicUsize::new(0));

    let short = df
        .lazy()
        .select([when(col("a").lt(lit(3)))
            .short_circuit()
            .then(counted_branch(short_first.clone(), 10))
            .when(col("a").lt(lit(5)))
            .then(counted_branch(short_second.clone(), 100))
            .otherwise(counted_branch(short_fallback.clone(), 1000))
            .alias("out")])
        .collect()?;

    assert_eq!(
        short
            .column("out")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![10, 20, 300, 400, 5000]
    );
    assert_eq!(short_first.load(Ordering::Relaxed), 2);
    assert_eq!(short_second.load(Ordering::Relaxed), 2);
    assert_eq!(short_fallback.load(Ordering::Relaxed), 1);

    Ok(())
}

#[test]
fn ternary_short_circuit_skips_unreachable_errors() -> PolarsResult<()> {
    fn exploding_branch() -> Expr {
        col("a").map(
            |_| polars_bail!(ComputeError: "should not run"),
            |_, f| Ok(f.clone()),
        )
    }

    let df = df!["a" => [1i32, 2, 3]]?;

    let short = df
        .clone()
        .lazy()
        .select([when(col("a").is_not_null())
            .short_circuit()
            .then(lit(1i32))
            .otherwise(exploding_branch())
            .alias("out")])
        .collect()?;
    assert_eq!(
        short
            .column("out")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![1, 1, 1]
    );

    let eager = df
        .lazy()
        .select([when(col("a").is_not_null())
            .then(lit(1i32))
            .otherwise(exploding_branch())
            .alias("out")])
        .collect();
    assert!(eager.is_err());

    Ok(())
}

#[test]
#[cfg(feature = "strings")]
fn includes_null_predicate_3038() -> PolarsResult<()> {
    let df = df! {
        "a" => [Some("a1"), None, None],
    }?;
    let res = df
        .lazy()
        .with_column(
            when(col("a").map(
                move |s| {
                    s.str()?
                        .to_lowercase()
                        .contains("not_exist", true)
                        .map(|ca| ca.into_column())
                },
                |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
            ))
            .then(lit("unexpected"))
            .when(col("a").eq(lit("a1".to_string())))
            .then(lit("good hit"))
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
            .alias("b"),
        )
        .collect()?;

    let exp_df = df! {
        "a" => [Some("a1"), None, None],
        "b" => [Some("good hit"), None, None],
    }?;
    assert!(res.equals_missing(&exp_df));

    let df = df! {
        "a" => ["a1", "a2", "a3", "a4", "a2"],
        "b" => [Some("tree"), None, None, None, None],
    }?;
    let res = df
        .lazy()
        .with_column(
            when(col("b").map(
                move |s| {
                    s.str()?
                        .to_lowercase()
                        .contains_literal("non-existent")
                        .map(|ca| ca.into_column())
                },
                |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
            ))
            .then(lit("weird-1"))
            .when(col("a").eq(lit("a1".to_string())))
            .then(lit("ok1"))
            .when(col("a").eq(lit("a2".to_string())))
            .then(lit("ok2"))
            .when(lit(true))
            .then(lit("ft"))
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
            .alias("c"),
        )
        .collect()?;
    let exp_df = df! {
        "a" => ["a1", "a2", "a3", "a4", "a2"],
        "b" => [Some("tree"), None, None, None, None],
        "c" => ["ok1", "ok2", "ft", "ft", "ok2"]
    }?;
    assert!(res.equals_missing(&exp_df));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_when_then_otherwise_cats() -> PolarsResult<()> {
    let lf = df!["book" => [Some("bookA"),
        None,
        Some("bookB"),
        None,
        Some("bookA"),
        Some("bookC"),
        Some("bookC"),
        Some("bookC")],
        "user" => [Some("bob"), Some("bob"), Some("bob"), Some("tim"), Some("lucy"), Some("lucy"), None, None]
    ]?.lazy();

    let out = lf
        .with_column(col("book").cast(DataType::from_categories(Categories::global())))
        .with_column(col("user").cast(DataType::from_categories(Categories::global())))
        .with_column(
            when(col("book").is_null())
                .then(col("user"))
                .otherwise(col("book"))
                .alias("a"),
        )
        .collect()?;

    assert_eq!(
        out.column("a")?
            .cat32()?
            .iter_str()
            .flatten()
            .collect::<Vec<_>>(),
        &[
            "bookA", "bob", "bookB", "tim", "bookA", "bookC", "bookC", "bookC"
        ]
    );

    Ok(())
}

#[test]
fn test_when_then_otherwise_single_bool() -> PolarsResult<()> {
    let df = df![
        "key" => ["a", "b", "b"],
        "val" => [Some(1), Some(2), None]
    ]?;

    let out = df
        .lazy()
        .group_by_stable([col("key")])
        .agg([when(col("val").null_count().gt(lit(0)))
            .then(Null {}.lit())
            .otherwise(col("val").sum())
            .alias("sum_null_prop")])
        .collect()?;

    let expected = df![
        "key" => ["a", "b"],
        "sum_null_prop" => [Some(1), None]
    ]?;

    assert!(out.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "unique_counts")]
fn test_update_groups_in_cast() -> PolarsResult<()> {
    let df = df![
        "group" =>  ["A" ,"A", "A", "B", "B", "B", "B"],
        "id"=> [1, 2, 1, 4, 5, 4, 6],
    ]?;

    // optimized to
    // col("id").unique_counts().cast(int64) * -1
    // in aggregation that cast coerces a list and the cast may forget to update groups
    let out = df
        .lazy()
        .group_by_stable([col("group")])
        .agg([col("id").unique_counts() * lit(-1)])
        .collect()?;

    let expected = df![
        "group" =>  ["A" ,"B"],
        "id"=> [AnyValue::List(Series::new("".into(), [-2i64, -1])), AnyValue::List(Series::new("".into(), [-2i64, -1, -1]))]
    ]?;

    assert!(out.equals(&expected));
    Ok(())
}

#[test]
fn test_when_then_otherwise_sum_in_agg() -> PolarsResult<()> {
    let df = df![
        "groups" => [1, 1, 2, 2],
        "dist_a" => [0.1, 0.2, 0.5, 0.5],
        "dist_b" => [0.8, 0.2, 0.5, 0.2],
    ]?;

    let q = df
        .lazy()
        .group_by([col("groups")])
        .agg([
            when(all().exclude_cols(["groups"]).as_expr().sum().eq(lit(1)))
                .then(all().exclude_cols(["groups"]).as_expr().sum())
                .otherwise(lit(LiteralValue::untyped_null())),
        ])
        .sort(["groups"], Default::default());

    let expected = df![
        "groups" => [1, 2],
        "dist_a" => [None, Some(1.0f64)],
        "dist_b" => [Some(1.0f64), None]
    ]?;
    assert!(q.collect()?.equals_missing(&expected));

    Ok(())
}

#[test]
fn test_binary_over_3930() -> PolarsResult<()> {
    let df = df![
        "class" => ["a", "a", "a", "b", "b", "b"],
        "score" => [0.2, 0.5, 0.1, 0.3, 0.4, 0.2]
    ]?;

    let ss = col("score").pow(2);
    let mdiff = (ss.clone().shift(lit(-1)) - ss.shift(lit(1))) / lit(2);
    let out = df
        .lazy()
        .select([mdiff.over([col("class")]).unwrap()])
        .collect()?;

    let out = out.column("score")?;
    let out = out.f64()?;

    assert_eq!(
        Vec::from(out),
        &[
            None,
            Some(-0.015000000000000003),
            None,
            None,
            Some(-0.024999999999999994),
            None
        ]
    );

    Ok(())
}

#[test]
#[cfg(feature = "rank")]
fn test_ternary_aggregation_set_literals() -> PolarsResult<()> {
    let df = df![
        "name" => ["a", "b", "a", "b"],
        "value" => [1, 3, 2, 4]
    ]?;

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(col("value").rank(Default::default(), None))
            .otherwise(lit(Series::new("".into(), &[10 as IdxSize])))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert_eq!(
        out.get(0)?,
        AnyValue::List(Series::new("".into(), &[1 as IdxSize, 2 as IdxSize]))
    );
    assert_eq!(
        out.get(1)?,
        AnyValue::List(Series::new("".into(), &[10 as IdxSize, 10 as IdxSize]))
    );

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(lit(Series::new("".into(), &[10 as IdxSize])).alias("value"))
            .otherwise(col("value").rank(Default::default(), None))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert_eq!(
        out.get(1)?,
        AnyValue::List(Series::new("".into(), &[1 as IdxSize, 2]))
    );
    assert_eq!(
        out.get(0)?,
        AnyValue::List(Series::new("".into(), &[10 as IdxSize, 10 as IdxSize]))
    );

    let out = df
        .clone()
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(col("value").rank(Default::default(), None))
            .otherwise(Null {}.lit())])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert!(matches!(out.get(0)?, AnyValue::List(_)));
    assert!(matches!(out.get(1)?, AnyValue::List(_)));

    // swapped branch
    let out = df
        .lazy()
        .group_by([col("name")])
        .agg([when(col("value").sum().eq(lit(3)))
            .then(Null {}.lit().alias("value"))
            .otherwise(col("value").rank(Default::default(), None))])
        .sort(["name"], Default::default())
        .collect()?;

    let out = out.column("value")?;
    assert!(matches!(out.get(1)?, AnyValue::List(_)));
    assert!(matches!(out.get(0)?, AnyValue::List(_)));

    Ok(())
}

#[test]
fn test_binary_group_consistency() -> PolarsResult<()> {
    let lf = df![
        "name" => ["a", "b", "c", "d"],
        "category" => [1, 2, 3, 4],
        "score" => [3, 5, 1, 2],
    ]?
    .lazy();

    let out = lf
        .group_by([col("category")])
        .agg([col("name").filter(col("score").eq(col("score").max()))])
        .sort(["category"], Default::default())
        .collect()?;
    let out = out.column("name")?;

    assert_eq!(out.dtype(), &DataType::List(Box::new(DataType::String)));
    assert_eq!(
        out.explode(ExplodeOptions {
            empty_as_null: true,
            keep_nulls: true
        })?
        .str()?
        .no_null_iter()
        .collect::<Vec<_>>(),
        &["a", "b", "c", "d"]
    );

    Ok(())
}
