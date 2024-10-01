use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::dsl::Expr;
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
fn test_median() {
    let expr = col("Sales").median();

    let sql_expr = "MEDIAN(Sales)";
    let (expected, actual) = create_expected(expr, sql_expr);

    assert!(expected.equals(&actual))
}

#[test]
fn test_quantile_cont() {
    for &q in &[0.25, 0.5, 0.75] {
        let expr = col("Sales").quantile(lit(q), QuantileInterpolOptions::Linear);

        let sql_expr = format!("QUANTILE_CONT(Sales, {})", q);
        let (expected, actual) = create_expected(expr, &sql_expr);

        assert!(
            expected.equals(&actual),
            "q: {q}: expected {expected:?}, got {actual:?}"
        )
    }
}
