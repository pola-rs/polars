use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_df() -> LazyFrame {
    df! {
      "Year" => [2018, 2018, 2019, 2019, 2020, 2020],
      "Country" => ["US", "UK", "US", "UK", "US", "UK"],
      "Sales" => [1000, 2000, 3000, 4000, 5000, 6000]
    }
    .unwrap()
    .lazy()
}

fn create_expected(expr: Expr, sql: &str) -> (DataFrame, DataFrame) {
    let df = create_df();
    let alias = "TEST";

    let query = format!(
        r#"
      SELECT
          {sql} as {alias}
      FROM
          df
      ORDER BY
        {alias}
      "#
    );

    let expected = df
        .clone()
        .select(&[expr.alias(alias)])
        .sort([alias], Default::default())
        .collect()
        .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let actual = ctx.execute(&query).unwrap().collect().unwrap();
    (expected, actual)
}

#[test]
fn test_cumulative_sum() {
    let expr = col("Sales")
        .sort(SortOptions::default().with_order_descending(true))
        .cum_sum(false);

    let sql_expr = "SUM(Sales) OVER (ORDER BY Sales DESC)";
    let (expected, actual) = create_expected(expr, sql_expr);

    assert!(expected.equals(&actual))
}

#[test]
fn test_cumulative_min() {
    let expr = col("Sales")
        .sort(SortOptions::default().with_order_descending(true))
        .cum_min(false);

    let sql_expr = "MIN(Sales) OVER (ORDER BY Sales DESC)";
    let (expected, actual) = create_expected(expr, sql_expr);

    assert!(expected.equals(&actual))
}

#[test]
fn test_cumulative_max() {
    let expr = col("Sales")
        .sort(SortOptions::default().with_order_descending(true))
        .cum_max(false);

    let sql_expr = "MAX(Sales) OVER (ORDER BY Sales DESC)";
    let (expected, actual) = create_expected(expr, sql_expr);

    assert!(expected.equals(&actual))
}
