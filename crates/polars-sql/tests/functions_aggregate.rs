use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::dsl::Expr;
use polars_sql::*;

fn create_df() -> LazyFrame {
    df! {
      "Data" => [1000, 2000, 3000, 4000, 5000, 6000]
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
    let expr = col("Data").median();

    let sql_expr = "MEDIAN(Data)";
    let (expected, actual) = create_expected(expr, sql_expr);

    assert!(expected.equals(&actual))
}

#[test]
fn test_quantile_cont() {
    for &q in &[0.25, 0.5, 0.75] {
        let expr = col("Data").quantile(lit(q), QuantileMethod::Linear);

        let sql_expr = format!("QUANTILE_CONT(Data, {})", q);
        let (expected, actual) = create_expected(expr, &sql_expr);

        assert!(
            expected.equals(&actual),
            "q: {q}: expected {expected:?}, got {actual:?}"
        )
    }
}

#[test]
fn test_quantile_disc() {
    for &q in &[0.25, 0.5, 0.75] {
        let expr = col("Data").quantile(lit(q), QuantileMethod::Equiprobable);

        let sql_expr = format!("QUANTILE_DISC(Data, {})", q);
        let (expected, actual) = create_expected(expr, &sql_expr);

        assert!(expected.equals(&actual))
    }
}

#[test]
fn test_quantile_out_of_range() {
    for &q in &["-1", "2", "-0.01", "1.01"] {
        for &func in &["QUANTILE_CONT", "QUANTILE_DISC"] {
            let query = format!("SELECT {func}(Data, {q})");
            let mut ctx = SQLContext::new();
            ctx.register("df", create_df());
            let actual = ctx.execute(&query);
            assert!(actual.is_err())
        }
    }
}

#[test]
fn test_quantile_disc_conformance() {
    let expected = df![
        "q" => [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Data" => [1000, 1000, 2000, 2000, 3000, 3000, 4000, 5000, 5000, 6000, 6000],
    ]
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", create_df());

    let mut actual: Option<DataFrame> = None;
    for &q in &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let res = ctx
            .execute(&format!(
                "SELECT {q}::float as q, QUANTILE_DISC(Data, {q}) as Data FROM df"
            ))
            .unwrap()
            .collect()
            .unwrap();
        actual = if let Some(df) = actual {
            Some(df.vstack(&res).unwrap())
        } else {
            Some(res)
        };
    }

    assert!(
        expected.equals(actual.as_ref().unwrap()),
        "expected {expected:?}, got {actual:?}"
    )
}
