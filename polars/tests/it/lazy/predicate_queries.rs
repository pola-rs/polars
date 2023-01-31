// used only if feature="is_in", feature="dtype-categorical"
#[allow(unused_imports)]
use polars_core::{with_string_cache, SINGLE_LOCK};

use super::*;

#[test]
fn test_predicate_after_renaming() -> PolarsResult<()> {
    let df = df![
        "foo" => [1, 2, 3],
        "bar" => [3, 2, 1]
    ]?
    .lazy()
    .rename(["foo", "bar"], ["foo2", "bar2"])
    .filter(col("foo2").eq(col("bar2")))
    .collect()?;

    let expected = df![
        "foo2" => [2],
        "bar2" => [2],
    ]?;
    assert!(df.frame_equal(&expected));

    Ok(())
}

#[test]
fn filter_true_lit() -> PolarsResult<()> {
    let df = df! {
        "a" => [Some(true), Some(false), None],
        "b" => ["1", "2", "3"]
    }?;
    let filter = col("a").eq(lit(true));
    let with_true = df.clone().lazy().filter(filter.clone()).collect()?;
    let with_not_true = df
        .clone()
        .lazy()
        .filter(not(filter.clone()))
        .with_predicate_pushdown(false)
        .with_projection_pushdown(false)
        .collect()?;
    let res = with_true.vstack(&with_not_true)?;
    assert!(res.frame_equal_missing(&df));
    Ok(())
}

#[test]
fn test_combine_columns_in_filter() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [None, Some("a"), Some("b")]
    ]?;

    let out = df
        .lazy()
        .filter(
            cols(vec!["a".to_string(), "b".to_string()])
                .cast(DataType::Utf8)
                .gt(lit("2")),
        )
        .collect()?;

    let expected = df![
        "a" => [3],
        "b" => ["b"],
    ]?;

    // "b" > "2" == true
    assert!(out.frame_equal(&expected));
    Ok(())
}

fn create_n_filters(col_name: &str, num_filters: usize) -> Vec<Expr> {
    (0..num_filters)
        .into_iter()
        .map(|i| col(col_name).eq(lit(format!("{}", i))))
        .collect()
}

fn and_filters(expr: Vec<Expr>) -> Expr {
    expr.into_iter().reduce(polars::prelude::Expr::and).unwrap()
}

#[test]
fn test_many_filters() -> PolarsResult<()> {
    // just check if it runs. in #3210
    // we had terrible tree traversion perf.
    let df = df! {
        "id" => ["1", "2"]
    }?;
    let filters = create_n_filters("id", 30);
    let _ = df
        .lazy()
        .filter(and_filters(filters))
        .with_predicate_pushdown(false)
        .collect()?;

    Ok(())
}

#[test]
fn test_filter_no_combine() -> PolarsResult<()> {
    let df = df![
        "vals" => [1, 2, 3, 4, 5]
    ]?;

    let out = df
        .lazy()
        .filter(col("vals").gt(lit(1)))
        // should be > 2
        // if optimizer would combine predicates this would be flawed
        .filter(col("vals").gt(col("vals").min()))
        .collect()?;

    assert_eq!(
        Vec::from(out.column("vals")?.i32()?),
        &[Some(3), Some(4), Some(5)]
    );

    Ok(())
}

#[test]
fn test_filter_block_join() -> PolarsResult<()> {
    let df_a = df![
        "a" => ["a", "b", "c"],
        "c" => [1, 4, 6]
    ]?;
    let df_b = df![
        "a" => ["a", "a", "c"],
        "d" => [2, 4, 3]
    ]?;

    let out = df_a
        .lazy()
        .left_join(df_b.lazy(), "a", "a")
        // mean is influence by join
        .filter(col("c").mean().eq(col("d")))
        .collect()?;
    assert_eq!(out.shape(), (1, 3));

    Ok(())
}

#[test]
#[cfg(all(feature = "is_in", feature = "dtype-categorical"))]
fn test_is_in_categorical_3420() -> PolarsResult<()> {
    let df = df![
        "a" => ["a", "b", "c", "d", "e"],
        "b" => [1, 2, 3, 4, 5]
    ]?;

    let _guard = SINGLE_LOCK.lock();

    let _: PolarsResult<_> = with_string_cache(|| {
        let s = Series::new("x", ["a", "b", "c"]).strict_cast(&DataType::Categorical(None))?;
        let out = df
            .lazy()
            .with_column(col("a").strict_cast(DataType::Categorical(None)))
            .filter(col("a").is_in(lit(s).alias("x")))
            .collect()?;

        let mut expected = df![
            "a" => ["a", "b", "c"],
            "b" => [1, 2, 3]
        ]?;
        expected.try_apply("a", |s| s.cast(&DataType::Categorical(None)))?;
        assert!(out.frame_equal(&expected));

        Ok(())
    });
    Ok(())
}

#[test]
fn test_predicate_pushdown_blocked_by_outer_join() -> PolarsResult<()> {
    let df1 = df! {
        "a" => ["a1", "a2"],
        "b" => ["b1", "b2"]
    }?;
    let df2 = df! {
        "b" => ["b2", "b3"],
        "c" => ["c2", "c3"]
    }?;
    let df = df1.lazy().outer_join(df2.lazy(), col("b"), col("b"));
    let out = df.clone().filter(col("a").eq(lit("a1"))).collect()?;
    let null: Option<&str> = None;
    let expected = df![
        "a" => ["a1"],
        "b" => ["b1"],
        "c" => [null],
    ]?;
    assert!(out.frame_equal_missing(&expected));
    Ok(())
}

#[test]
fn test_count_blocked_at_union_3963() -> PolarsResult<()> {
    let lf1 = df![
        "k" => ["x", "x", "y"],
        "v" => [3, 2, 6,]
    ]?
    .lazy();

    let lf2 = df![
        "k" => ["a", "a", "b"],
        "v" => [1, 8, 5]
    ]?
    .lazy();

    let expected = df![
        "k" => ["x", "x", "a", "a"],
        "v" => [3, 2, 1, 8]
    ]?;

    for rechunk in [true, false] {
        let out = concat([lf1.clone(), lf2.clone()], rechunk, true)?
            .filter(count().over([col("k")]).gt(lit(1)))
            .collect()?;

        assert!(out.frame_equal(&expected));
    }

    Ok(())
}

#[test]
fn test_predicate_on_join_select_4884() -> PolarsResult<()> {
    let lf = df![
      "x" => [0, 1],
      "y" => [1, 2],
    ]?
    .lazy();
    let out = (lf.clone().join_builder().with(lf))
        .left_on([col("y")])
        .right_on([col("x")])
        .suffix("_right")
        .finish()
        .select([col("x"), col("y_right").alias("y")])
        .filter(col("x").neq(col("y")).and(col("y").eq(2)))
        .collect()?;

    let expected = df![
      "x" => [0],
      "y" => [2],
    ]?;
    assert_eq!(out, expected);
    Ok(())
}
