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
fn test_exists_lt_correlation() -> PolarsResult<()> {
    let df = run("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?;
    let expected = df! { "a" => [2i64, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_not_exists_lt_correlation() -> PolarsResult<()> {
    let df = run("SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?;
    let expected = df! { "a" => [1i64] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_gt_and_gte_and_lte_correlation() -> PolarsResult<()> {
    let df = run("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS x WHERE x.b > t1.b)")?;
    let expected = df! { "a" => [1i64, 2] }?;
    assert!(df.equals(&expected), "got {df:?}");

    let df = run("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS x WHERE x.b >= t1.b)")?;
    let expected = df! { "a" => [1i64, 2, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");

    let df = run("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS x WHERE x.b <= t1.b)")?;
    let expected = df! { "a" => [1i64, 2, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_neq_correlation() -> PolarsResult<()> {
    let df = run("SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t1 AS x WHERE x.b <> t1.b)")?;
    let expected = df! { "a" => [1i64, 2, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn test_equality_correlated_exists_still_uses_semi_join() -> PolarsResult<()> {
    let sql = "SELECT a FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)";
    let plan = ctx().execute(sql)?.explain(true)?;
    assert!(plan.contains("SEMI JOIN"), "{plan}");

    let df = run(sql)?;
    let expected = df! { "a" => [1i64, 2] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn test_equality_correlated_not_exists_still_uses_anti_join() -> PolarsResult<()> {
    let sql = "SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)";
    let plan = ctx().execute(sql)?.explain(true)?;
    assert!(plan.contains("ANTI JOIN"), "{plan}");

    let df = run(sql)?;
    let expected = df! { "a" => [3i64] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_no_matches_at_all() -> PolarsResult<()> {
    // No `x.b` value can be both less than and greater than the same
    // `t1.b`, so EXISTS is false for every outer row.
    let df = run("SELECT a FROM t1 WHERE EXISTS \
         (SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b)")?;
    assert_eq!(df.height(), 0);
    Ok(())
}

#[test]
fn test_not_exists_all_rows_match() -> PolarsResult<()> {
    // Complement of the always-false EXISTS above: NOT EXISTS is true for
    // every outer row.
    let df = run("SELECT a FROM t1 WHERE NOT EXISTS \
         (SELECT 1 FROM t1 AS x WHERE x.b < t1.b AND x.b > t1.b)")?;
    let expected = df! { "a" => [1i64, 2, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_inequality_combined_with_other_where_conjunct() -> PolarsResult<()> {
    let df =
        run("SELECT a FROM t1 WHERE a > 1 AND EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?;
    let expected = df! { "a" => [2i64, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_inequality_with_inner_local_filter() -> PolarsResult<()> {
    // `t2.w > 1` filters the inner relation to (g=10, w=2) and (g=20, w=3)
    // before the `<` correlation is evaluated.
    let df = run("SELECT a FROM t1 WHERE EXISTS \
         (SELECT 1 FROM t2 WHERE t2.g < t1.b AND t2.w > 1)")?;
    let expected = df! { "a" => [2i64, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_not_exists_inequality_null_edge_case() -> PolarsResult<()> {
    let t1 = df! {
        "a" => [1i64, 2, 3],
        "b" => [Some(10i64), None, Some(30)],
    }
    .unwrap()
    .lazy();
    let mut ctx = SQLContext::new();
    ctx.register("t1", t1);
    // Outer row `a=2` has a NULL `b`; every comparison against it is NULL (not
    // true), so its subquery matches zero rows and NOT EXISTS is true.
    let df = ctx
        .execute("SELECT a FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?
        .collect()?
        .sort(["a"], Default::default())?;
    let expected = df! { "a" => [1i64, 2] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

// --- EXISTS in general expression position (OR / CASE / SELECT list) -------
//
// None of these shapes are a whole WHERE filter or a top-level AND-conjunct
// of it, so `rewrite_subquery_conjuncts` can't intercept them; they exercise
// `decorrelate_exists_subqueries` and the `SQLExpr::Exists` arm of the
// expression visitor instead. Deliberately unguarded by the
// `semi_anti_join` feature: this lowering doesn't build semi/anti joins.

#[test]
fn test_exists_or_predicate() -> PolarsResult<()> {
    // `a = 99` never matches, so the result is exactly the EXISTS-true rows.
    let df = run("SELECT a FROM t1 WHERE a = 99 \
         OR EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?;
    let expected = df! { "a" => [2i64, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_not_exists_or_predicate() -> PolarsResult<()> {
    // `a = 99` never matches, so the result is exactly the NOT-EXISTS-true rows.
    let df = run("SELECT a FROM t1 WHERE a = 99 \
         OR NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b)")?;
    let expected = df! { "a" => [1i64] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_equality_correlated_exists_in_or_position() -> PolarsResult<()> {
    // An equality correlation, but not a top-level AND-conjunct, so this must
    // go through the general decorrelation path rather than the semi-join
    // fast path.
    let df = run("SELECT a FROM t1 WHERE a = 99 \
         OR EXISTS (SELECT 1 FROM t2 WHERE t2.g = t1.b)")?;
    let expected = df! { "a" => [1i64, 2] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_in_case_expression() -> PolarsResult<()> {
    let df = ctx()
        .execute(
            "SELECT a, CASE WHEN EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) \
             THEN 1 ELSE 0 END AS flag FROM t1",
        )?
        .collect()?
        .sort(["a"], Default::default())?;
    let expected = df! { "a" => [1i64, 2, 3], "flag" => [0i32, 1, 1] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_exists_in_select_list() -> PolarsResult<()> {
    let df = ctx()
        .execute("SELECT a, EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e FROM t1")?
        .collect()?
        .sort(["a"], Default::default())?;
    let expected = df! { "a" => [1i64, 2, 3], "e" => [false, true, true] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_not_exists_in_select_list() -> PolarsResult<()> {
    let df = ctx()
        .execute("SELECT a, NOT EXISTS (SELECT 1 FROM t1 AS x WHERE x.b < t1.b) AS e FROM t1")?
        .collect()?
        .sort(["a"], Default::default())?;
    let expected = df! { "a" => [1i64, 2, 3], "e" => [true, false, false] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}

#[test]
fn test_uncorrelated_exists_in_expression_position() -> PolarsResult<()> {
    // Uncorrelated EXISTS is a constant boolean, broadcast onto every row.
    let df = run("SELECT a FROM t1 WHERE a = 1 OR EXISTS (SELECT 1 FROM t2)")?;
    let expected = df! { "a" => [1i64, 2, 3] }?;
    assert!(df.equals(&expected), "got {df:?}");

    let df = run("SELECT a FROM t1 WHERE a = 1 OR EXISTS (SELECT 1 FROM t2 WHERE g > 1000)")?;
    let expected = df! { "a" => [1i64] }?;
    assert!(df.equals(&expected), "got {df:?}");
    Ok(())
}
