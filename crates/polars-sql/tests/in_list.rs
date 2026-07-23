use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_df() -> DataFrame {
    df! {
        "a" => [1, 2, 3, 4],
        "b" => [Some(2), Some(4), None, Some(10)],
    }
    .unwrap()
}

fn run(sql: &str) -> PolarsResult<DataFrame> {
    let mut ctx = SQLContext::new();
    ctx.register("nums", create_df().lazy());
    ctx.execute(sql)?.collect()
}

#[test]
fn test_in_list_column_elements() -> PolarsResult<()> {
    // `a IN (b - 1, b)`: purely column-derived elements (no literals).
    let df = run("SELECT a FROM nums WHERE a IN (b - 1, b) ORDER BY a")?;
    let expected = df! { "a" => [1] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

#[test]
fn test_in_list_arithmetic_elements() -> PolarsResult<()> {
    // `a IN (1 + 1, 3 * 3)`: constant-folding-shaped arithmetic elements
    // still go through the fallback (BinaryOp is not accepted by the
    // literal-array fast path).
    let df = run("SELECT a FROM nums WHERE a IN (1 + 1, 3 * 3) ORDER BY a")?;
    let expected = df! { "a" => [2] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

#[test]
fn test_in_list_mixed_literal_and_column() -> PolarsResult<()> {
    // `a IN (b - 2, 100)`: one column-derived element, one literal.
    let df = run("SELECT a FROM nums WHERE a IN (b - 2, 100) ORDER BY a")?;
    let expected = df! { "a" => [2] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

#[test]
fn test_in_list_not_in_with_column_elements() -> PolarsResult<()> {
    // NOT IN variant of the mixed case above; row with NULL b (a=3) stays
    // excluded because the membership test is unknown, not FALSE.
    let df = run("SELECT a FROM nums WHERE a NOT IN (b - 1, 100) ORDER BY a")?;
    let expected = df! { "a" => [2, 4] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

#[test]
fn test_in_list_all_literal_fast_path_unaffected() -> PolarsResult<()> {
    // Regression: an all-literal list must still go through the imploded-Series
    // `is_in` fast path (load-bearing for predicate pushdown), not the OR-chain.
    let mut ctx = SQLContext::new();
    ctx.register("nums", create_df().lazy());
    let lf = ctx.execute("SELECT a FROM nums WHERE a IN (1, 2, 4)")?;
    let plan = format!("{:?}", lf.clone().logical_plan);
    assert!(
        plan.contains("is_in"),
        "expected all-literal IN list to lower to `is_in`, got plan: {plan}"
    );
    assert!(
        !plan.contains(" or "),
        "expected all-literal IN list to avoid the OR-chain fallback, got plan: {plan}"
    );

    let df = lf.collect()?.sort(["a"], Default::default())?;
    let expected = df! { "a" => [1, 2, 4] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

#[test]
fn test_in_list_mixed_still_produces_correct_rows_for_larger_list() -> PolarsResult<()> {
    // Sanity check with a longer, fully mixed list (literal + column + arithmetic).
    let df = run("SELECT a FROM nums WHERE a IN (b - 1, 100, b * 2) ORDER BY a")?;
    // row1: a=1, b=2 -> {1, 100, 4}  -> match (1)
    // row2: a=2, b=4 -> {3, 100, 8}  -> no match
    // row3: a=3, b=NULL -> {NULL, 100, NULL} -> unknown (excluded)
    // row4: a=4, b=10 -> {9, 100, 20} -> no match
    let expected = df! { "a" => [1] }?;
    assert!(df.equals(&expected), "got {df:?}, expected {expected:?}");
    Ok(())
}

fn three_valued_case(sql_expr: &str) -> PolarsResult<Option<bool>> {
    let df = df! { "x" => [1] }?.lazy();
    let mut ctx = SQLContext::new();
    ctx.register("t", df);
    let out = ctx
        .execute(&format!("SELECT {sql_expr} AS r FROM t"))?
        .collect()?;
    let s = out.column("r")?.bool()?;
    Ok(s.get(0))
}

#[test]
fn test_in_list_null_three_valued_logic() -> PolarsResult<()> {
    // `1 IN (2, NULL)` -> unknown (NULL), not FALSE.
    assert_eq!(three_valued_case("1 IN (2, NULL)")?, None);
    // `1 NOT IN (2, NULL)` -> unknown (NULL).
    assert_eq!(three_valued_case("1 NOT IN (2, NULL)")?, None);
    // `1 IN (1, NULL)` -> TRUE (an actual match wins over the NULL element).
    assert_eq!(three_valued_case("1 IN (1, NULL)")?, Some(true));
    // `1 NOT IN (1, NULL)` -> FALSE.
    assert_eq!(three_valued_case("1 NOT IN (1, NULL)")?, Some(false));
    // NULL on the left-hand side is always unknown.
    assert_eq!(three_valued_case("(1 * NULL) IN (1, 2)")?, None);
    assert_eq!(three_valued_case("(1 * NULL) IN (x, 2)")?, None);
    Ok(())
}

#[test]
fn test_in_list_null_three_valued_logic_with_column_elements() -> PolarsResult<()> {
    // Same NULL semantics, but with a non-literal (column) element list so the
    // OR-chain fallback is exercised instead of the literal fast path.
    let df = df! {
        "x" => [1, 1, 1],
        "y" => [2, 1, 2],
        "z" => [Option::<i32>::None, None, None],
    }?
    .lazy();
    let mut ctx = SQLContext::new();
    ctx.register("t", df);
    let out = ctx.execute("SELECT x IN (y, z) AS r FROM t")?.collect()?;
    let s = out.column("r")?.bool()?;
    // row0: x=1, y=2, z=NULL -> no match, set has NULL -> unknown
    assert_eq!(s.get(0), None);
    // row1: x=1, y=1, z=NULL -> match on y -> TRUE
    assert_eq!(s.get(1), Some(true));
    // row2: x=1, y=2, z=NULL -> same as row0 -> unknown
    assert_eq!(s.get(2), None);
    Ok(())
}
