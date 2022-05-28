use super::*;
use polars_core::{with_string_cache, SINGLE_LOCK};

#[test]
fn test_predicate_after_renaming() -> Result<()> {
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
fn filter_true_lit() -> Result<()> {
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
fn test_combine_columns_in_filter() -> Result<()> {
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
fn test_many_filters() -> Result<()> {
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
fn test_filter_no_combine() -> Result<()> {
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
fn test_filter_block_join() -> Result<()> {
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
fn test_is_in_categorical_3420() -> Result<()> {
    let df = df![
        "a" => ["a", "b", "c", "d", "e"],
        "b" => [1, 2, 3, 4, 5]
    ]?;

    let _guard = SINGLE_LOCK.lock();

    let out: Result<_> = with_string_cache(|| {
        let s = Series::new("x", ["a", "b", "c"]).strict_cast(&DataType::Categorical(None))?;
        df.lazy()
            .with_column(col("a").strict_cast(DataType::Categorical(None)))
            .filter(col("a").is_in(lit(s).alias("x")))
            .collect()
    });
    let mut expected = df![
        "a" => ["a", "b", "c"],
        "b" => [1, 2, 3]
    ]?;
    expected.try_apply("a", |s| s.cast(&DataType::Categorical(None)))?;
    assert!(out?.frame_equal(&expected));
    Ok(())
}
