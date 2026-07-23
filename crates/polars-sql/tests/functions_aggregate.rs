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

        let sql_expr = format!("QUANTILE_CONT(Data, {q})");
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

        let sql_expr = format!("QUANTILE_DISC(Data, {q})");
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

fn create_df_corr() -> LazyFrame {
    df! {
        "a" => [1, 2, 3, 4, 5, 6],
        "b" => [2, 4, 10, 8, 9, 13],
        "c" => ["a", "b", "a", "a", "b", "b"]
    }
    .unwrap()
    .lazy()
}

#[test]
fn test_corr() {
    let df = create_df_corr();

    let expr_corr = pearson_corr(col("a"), col("b")).alias("corr");
    let expr_cov = cov(col("a"), col("b"), 1).alias("cov");
    let expr_cov_pop = cov(col("a"), col("b"), 0).alias("cov_pop");
    let expected = df
        .clone()
        .select(&[expr_corr, expr_cov, expr_cov_pop])
        .collect()
        .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let sql = r#"
    SELECT
        CORR(a, b) as corr,
        COVAR(a, b) as covar,
        COVAR_POP(a, b) as covar_pop
    FROM df"#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_corr_group_by() {
    let df = create_df_corr();

    let expected = df
        .clone()
        .group_by(["c"])
        .agg([
            pearson_corr(col("a"), col("b")).alias("corr"),
            cov(col("a"), col("b"), 1).alias("cov"),
        ])
        .sort(["c"], Default::default())
        .collect()
        .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let sql = r#"
    SELECT
        c,
        CORR(a, b) AS corr,
        COVAR(a, b) AS covar
    FROM df
    GROUP BY c
    ORDER BY c"#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_count_dtype_and_negation() {
    let df = df! { "a" => [1, 2, 3] }.unwrap().lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    let actual = ctx
        .execute("SELECT COUNT(*) AS c FROM df")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(actual.column("c").unwrap().dtype(), &DataType::Int64);

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT -COUNT(*) AS c FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "c" => [-3i64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_literal() {
    let df = df! { "a" => [1, 2, 3] }.unwrap().lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(5) AS s, SUM(-3) AS t FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => [15i64], "t" => [-9i64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_literal_empty_table() {
    let df = df! { "a" => Vec::<i32>::new() }.unwrap().lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(5) AS s FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => [Option::<i64>::None] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_min_max_distinct_is_noop() {
    let df = df! { "a" => [Some(3), Some(1), Some(1), None::<i32>, Some(3)] }
        .unwrap()
        .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    let actual = ctx
        .execute("SELECT MIN(DISTINCT a) AS mn, MAX(DISTINCT a) AS mx FROM df")
        .unwrap()
        .collect()
        .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let expected = ctx
        .execute("SELECT MIN(a) AS mn, MAX(a) AS mx FROM df")
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_avg_distinct_dedup() {
    let df = df! { "a" => [Some(1), Some(1), Some(2), Some(2), Some(3), None::<i32>] }
        .unwrap()
        .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(DISTINCT a) AS s, AVG(DISTINCT a) AS m FROM df")
        .unwrap()
        .collect()
        .unwrap();
    // distinct non-null values: {1, 2, 3} -> sum 6, avg 2.0
    let expected = df! { "s" => [6i32], "m" => [2.0f64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_distinct_literal() {
    // a broadcast literal has a single distinct value, so this must NOT hit the
    // `(literal * len())` fast-path that plain `SUM(<literal>)` takes.
    let df = df! { "a" => [1, 2, 3] }.unwrap().lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(DISTINCT 5) AS s FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => [5i64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_avg_distinct_empty_group() {
    let df = df! { "a" => Vec::<i32>::new() }.unwrap().lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(DISTINCT a) AS s, AVG(DISTINCT a) AS m FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => [Option::<i32>::None], "m" => [Option::<f64>::None] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_total_distinct_and_dtype() {
    let df = df! { "a" => [Some(1), Some(1), Some(2), Some(2), Some(3), None::<i32>] }
        .unwrap()
        .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    let actual = ctx
        .execute("SELECT SUM(DISTINCT a) AS s, TOTAL(DISTINCT a) AS t FROM df")
        .unwrap()
        .collect()
        .unwrap();
    // distinct non-null values: {1, 2, 3} -> sum 6, total 6.0
    let expected = df! { "s" => [6i32], "t" => [6.0f64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");

    // TOTAL always returns a Float64, even over an all-null input.
    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT TOTAL(a) AS t, TOTAL(DISTINCT a) AS td FROM df WHERE a IS NULL")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "t" => [0.0f64], "td" => [0.0f64] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
    assert_eq!(actual.column("t").unwrap().dtype(), &DataType::Float64);
}

#[test]
fn test_group_concat_distinct_with_separator_errors() {
    let df = df! { "a" => [1, 1, 2] }.unwrap().lazy();

    // DISTINCT + explicit separator: not supported (matches SQLite's
    // "DISTINCT aggregates must have exactly one argument").
    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    assert!(
        ctx.execute("SELECT GROUP_CONCAT(DISTINCT a, ':') AS s FROM df")
            .is_err()
    );

    // DISTINCT with a single argument (no separator) must still work.
    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    let actual = ctx
        .execute("SELECT GROUP_CONCAT(DISTINCT a) AS s FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => ["1,2"] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");

    // Non-DISTINCT with an explicit separator must still work.
    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT GROUP_CONCAT(a, ':') AS s FROM df")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! { "s" => ["1:1:2"] }.unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}

#[test]
fn test_sum_min_max_empty_and_null_group() {
    let df = df! {
        "g" => ["a", "a", "b", "b"],
        "v" => [Some(10), Some(20), None::<i32>, None],
    }
    .unwrap()
    .lazy();

    let mut ctx = SQLContext::new();
    ctx.register("df", df.clone());
    let actual = ctx
        .execute(
            "SELECT g, SUM(v) AS s, MIN(v) AS mn, MAX(v) AS mx \
             FROM df GROUP BY g ORDER BY g",
        )
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! {
        "g" => ["a", "b"],
        "s" => [Some(30), None],
        "mn" => [Some(10), None],
        "mx" => [Some(20), None],
    }
    .unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    let actual = ctx
        .execute("SELECT SUM(v) AS s, MIN(v) AS mn, MAX(v) AS mx FROM df WHERE v > 1000")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df! {
        "s" => [Option::<i32>::None],
        "mn" => [Option::<i32>::None],
        "mx" => [Option::<i32>::None],
    }
    .unwrap();
    assert_eq!(expected, actual, "expected {expected:?}, got {actual:?}");
}
