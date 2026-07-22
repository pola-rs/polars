use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_df() -> DataFrame {
    df! {
        "x" => [1, 2, 3, 4, 5],
        "y" => [10, 20, 30, 40, 50],
    }
    .unwrap()
}

fn run(sql: &str) -> PolarsResult<DataFrame> {
    let mut ctx = SQLContext::new();
    ctx.register("df", create_df().lazy());
    ctx.execute(sql)?.collect()
}

#[test]
fn test_scalar_subquery_comparison_operators() -> PolarsResult<()> {
    let cases = [
        ("=", vec![5]),
        ("!=", vec![1, 2, 3, 4]),
        ("<", vec![1, 2, 3, 4]),
        ("<=", vec![1, 2, 3, 4, 5]),
        (">", vec![]),
        (">=", vec![5]),
    ];
    for (op, expected) in cases {
        let df = run(&format!(
            "SELECT x FROM df WHERE x {op} (SELECT MAX(x) FROM df)"
        ))?
        .sort(["x"], Default::default())?;
        let expected = df! { "x" => expected.into_iter().map(|v: i32| v).collect::<Vec<_>>() }?;
        assert!(
            df.equals(&expected),
            "op {op}: got {df:?}, expected {expected:?}"
        );
    }
    Ok(())
}

#[test]
fn test_scalar_subquery_on_left_hand_side() -> PolarsResult<()> {
    let df = run("SELECT x FROM df WHERE (SELECT MAX(x) FROM df) > x")?
        .sort(["x"], Default::default())?;
    let expected = df! { "x" => [1, 2, 3, 4] }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_scalar_subquery_arithmetic_operand() -> PolarsResult<()> {
    let df = run("SELECT x, x + (SELECT MAX(x) FROM df) AS z FROM df")?
        .sort(["x"], Default::default())?;
    let expected = df! {
        "x" => [1, 2, 3, 4, 5],
        "z" => [6, 7, 8, 9, 10],
    }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_scalar_subquery_in_select_list() -> PolarsResult<()> {
    let df =
        run("SELECT x, (SELECT MAX(x) FROM df) AS mx FROM df")?.sort(["x"], Default::default())?;
    let expected = df! {
        "x" => [1, 2, 3, 4, 5],
        "mx" => [5, 5, 5, 5, 5],
    }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_scalar_subquery_in_having() -> PolarsResult<()> {
    let df = run("SELECT x, SUM(y) AS s FROM df GROUP BY x \
         HAVING SUM(y) > (SELECT AVG(y) FROM df)")?
    .sort(["x"], Default::default())?;
    let expected = df! {
        "x" => [4, 5],
        "s" => [40, 50],
    }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_scalar_subquery_with_own_cte() -> PolarsResult<()> {
    let df = run("SELECT x FROM df WHERE x = \
         (WITH cte AS (SELECT MAX(x) AS m FROM df) SELECT m FROM cte)")?;
    let expected = df! { "x" => [5] }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_scalar_subquery_zero_rows_is_null() -> PolarsResult<()> {
    let df = run("SELECT x FROM df WHERE x = (SELECT x FROM df WHERE x > 100)")?;
    assert_eq!(df.height(), 0);
    Ok(())
}

#[test]
fn test_scalar_subquery_multi_row_takes_first() -> PolarsResult<()> {
    // PostgreSQL semantics: a scalar subquery returning more than one row is a
    // runtime error. Polars does not currently enforce that (would need engine
    // support to check safely under the streaming executor); it silently takes
    // the first row instead, matching the existing WHERE-clause `SubPlan` rewrite.
    let df = run("SELECT x FROM df WHERE x = (SELECT x FROM df ORDER BY x)")?;
    let expected = df! { "x" => [1] }?;
    assert!(df.equals(&expected));
    Ok(())
}

#[test]
fn test_unaliased_scalar_subquery_in_select_list() -> PolarsResult<()> {
    let df = run("SELECT (SELECT MAX(x) FROM df) FROM df")?;
    assert_eq!(df.height(), 5);
    let name = df.get_column_names_owned()[0].clone();
    let col = df.column(name.as_str())?.as_materialized_series().clone();
    let expected = Series::new(name, [5, 5, 5, 5, 5]);
    assert!(col.equals(&expected));
    Ok(())
}
