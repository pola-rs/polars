use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn ctx() -> SQLContext {
    let t1 = df! {
        "a" => [1i64, 2, 3],
        "b" => [10i64, 20, 30],
    }
    .unwrap()
    .lazy();
    let t2 = df! {
        "g" => [10i64, 10, 20],
        "w" => [1i64, 2, 3],
    }
    .unwrap()
    .lazy();
    let ctx = SQLContext::new();
    ctx.register("t1", t1);
    ctx.register("t2", t2);
    ctx
}

fn run(sql: &str) -> PolarsResult<DataFrame> {
    ctx()
        .execute(sql)?
        .collect()?
        .sort(["a"], Default::default())
}

#[test]
fn test_correlated_count_inequality() -> PolarsResult<()> {
    let df = run(
        "SELECT a, CAST((SELECT count(*) FROM t1 AS x WHERE x.b < t1.b) AS BIGINT) AS cnt FROM t1",
    )?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "cnt" => [0i64, 1, 2],
    }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_correlated_sum_empty_match_is_null() -> PolarsResult<()> {
    let df = run("SELECT a, (SELECT SUM(x.b) FROM t1 AS x WHERE x.b < t1.b) AS s FROM t1")?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "s" => [None, Some(10i64), Some(30)],
    }?;
    assert!(df.equals_missing(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_correlated_min_max_avg() -> PolarsResult<()> {
    let df = run("SELECT a, \
            (SELECT MIN(x.b) FROM t1 AS x WHERE x.b < t1.b) AS mn, \
            (SELECT MAX(x.b) FROM t1 AS x WHERE x.b < t1.b) AS mx, \
            (SELECT AVG(x.b) FROM t1 AS x WHERE x.b < t1.b) AS av \
         FROM t1")?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "mn" => [None, Some(10i64), Some(10)],
        "mx" => [None, Some(10i64), Some(20)],
        "av" => [None, Some(10.0f64), Some(15.0)],
    }?;
    assert!(df.equals_missing(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_correlated_equality_across_tables() -> PolarsResult<()> {
    let df =
        run("SELECT a, CAST((SELECT COUNT(*) FROM t2 WHERE t2.g = t1.b) AS BIGINT) AS c FROM t1")?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "c" => [2i64, 1, 0],
    }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_correlated_subquery_in_where() -> PolarsResult<()> {
    let df = run("SELECT a FROM t1 WHERE (SELECT COUNT(*) FROM t1 AS x WHERE x.b < t1.b) > 0")?;
    let expected = df! { "a" => [2i64, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_multiple_correlated_subqueries() -> PolarsResult<()> {
    let df = run("SELECT a, \
            CAST((SELECT COUNT(*) FROM t1 AS x WHERE x.b < t1.b) AS BIGINT) AS c, \
            (SELECT SUM(x.b) FROM t1 AS x WHERE x.b < t1.b) AS s \
         FROM t1")?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "c" => [0i64, 1, 2],
        "s" => [None, Some(10i64), Some(30)],
    }?;
    assert!(df.equals_missing(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_correlated_count_equality_self() -> PolarsResult<()> {
    // Self-correlated equality: count rows of the same table sharing `b`,
    // excluding the row itself via an inequality on `a`.
    let df = run(
        "SELECT a, CAST((SELECT COUNT(*) FROM t1 AS x WHERE x.b = t1.b AND x.a <> t1.a) AS BIGINT) AS c FROM t1",
    )?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "c" => [0i64, 0, 0],
    }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_uncorrelated_scalar_subquery_still_works() -> PolarsResult<()> {
    // An uncorrelated scalar subquery must NOT be routed through the
    // decorrelation path; it stays on the generic scalar-subquery path.
    let df = run("SELECT a, (SELECT MAX(b) FROM t1) AS mx FROM t1")?;
    let expected = df! {
        "a" => [1i64, 2, 3],
        "mx" => [30i64, 30, 30],
    }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}
