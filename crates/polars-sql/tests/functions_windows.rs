use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::dsl::Expr;
use polars_sql::*;

fn create_df() -> LazyFrame {
    df! {
      "a" => [1, 1, 1, 2, 2, 3],
      "b" => ["a", "b", "c", "a", "b", "c"]
    }
    .unwrap()
    .lazy()
}

fn create_expected(exprs: &[Expr], sql: &str) -> (DataFrame, DataFrame) {
    let df = create_df();

    let query = format!(
        r#"
      SELECT
          {sql}
      FROM
          df
      "#
    );

    let expected = df.clone().select(exprs).collect().unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let actual = ctx.execute(&query).unwrap().collect().unwrap();
    (expected, actual)
}

fn ensure_error(sql: &str, expected_error: &str) {
    let df = create_df();
    let query = format!(
        r#"
      SELECT
          {sql}
      FROM
          df
      "#
    );

    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    match ctx.execute(&query) {
        Ok(_) => panic!("expected error: {expected_error}"),
        Err(e) => {
            assert!(
                e.to_string().contains(expected_error),
                "expected error: {expected_error}, got: {e}",
            )
        },
    };
}

#[test]
fn test_lead_lag() {
    for shift in [-2, -1, 1, 2] {
        let (sql_func, sql_shift) = if shift > 0 {
            ("LAG", shift)
        } else {
            ("LEAD", -shift)
        };
        let exprs = [
            col("a"),
            col("b"),
            col("b")
                .shift(shift.into())
                .over_with_options(
                    Some([col("a")]),
                    Some(([col("b")], SortOptions::new().with_order_descending(false))),
                    Default::default(),
                )
                .unwrap()
                .alias("c"),
        ];

        let sql_expr =
            format!("a, b, {sql_func}(b, {sql_shift}) OVER (PARTITION BY a ORDER BY b) as c");
        let (expected, actual) = create_expected(&exprs, &sql_expr);

        assert_eq!(expected, actual, "shift: {shift}");
    }
}

#[test]
fn test_lead_lag_default() {
    for shift in [-1, 1] {
        let sql_func = if shift > 0 { "LAG" } else { "LEAD" };
        let exprs = [
            col("a"),
            col("b"),
            col("b")
                .shift(shift.into())
                .over_with_options(
                    Some([col("a")]),
                    Some(([col("b")], SortOptions::new().with_order_descending(false))),
                    Default::default(),
                )
                .unwrap()
                .alias("c"),
        ];

        let sql_expr = format!("a, b, {sql_func}(b) OVER (PARTITION BY a ORDER BY b) as c");
        let (expected, actual) = create_expected(&exprs, &sql_expr);

        assert_eq!(expected, actual, "shift: {shift}");
    }
}

#[test]
fn test_incorrect_shift() {
    for func in ["LAG", "LEAD"] {
        // Type of second argument is not an integer
        ensure_error(
            &format!("a, b, {func}(b, '1') OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be an integer",
        );
        ensure_error(
            &format!("a, b, {func}(b, 1.0) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be an integer",
        );
        ensure_error(
            &format!("a, b, {func}(b, 1.0) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be an integer",
        );

        // Number of arguments is incorrect
        ensure_error(
            &format!("a, b, {func}() OVER (PARTITION BY a ORDER BY b) as c"),
            "expects 1 or 2 arguments",
        );
        ensure_error(
            &format!("a, b, {func}(b, 1, 2) OVER (PARTITION BY a ORDER BY b) as c"),
            "expects 1 or 2 arguments",
        );

        // Second argument is not a constant
        ensure_error(
            &format!("a, b, {func}(b, a) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be an integer",
        );
        ensure_error(
            &format!("a, b, {func}(b, a + 1) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be an integer",
        );

        // Second argument is not positive
        ensure_error(
            &format!("a, b, {func}(b, -1) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be positive",
        );
        ensure_error(
            &format!("a, b, {func}(b, 0) OVER (PARTITION BY a ORDER BY b) as c"),
            "offset must be positive",
        );
    }
}

#[test]
fn test_lead_lag_without_partition_by() {
    // Test LAG/LEAD with ORDER BY but without PARTITION BY
    // This should work correctly over the entire dataset
    for shift in [-1, 1] {
        let (sql_func, shift_value) = if shift > 0 {
            ("LAG", shift)
        } else {
            ("LEAD", -shift)
        };

        let exprs = [
            col("a"),
            col("b"),
            col("a")
                .shift(shift.into())
                .over_with_options(
                    None, // No partition by
                    Some(([col("a")], SortOptions::new().with_order_descending(false))),
                    Default::default(),
                )
                .unwrap()
                .alias("a_shifted"),
        ];

        let sql_expr = format!("a, b, {sql_func}(a, {shift_value}) OVER (ORDER BY a) as a_shifted");
        let (expected, actual) = create_expected(&exprs, &sql_expr);

        assert_eq!(expected, actual, "shift: {shift}");
    }
}

#[test]
fn test_lead_lag_without_over_clause() {
    // LAG/LEAD without OVER clause should raise an error
    for func in ["LAG", "LEAD"] {
        ensure_error(&format!("a, b, {func}(b) as c"), "requires an OVER clause");
        ensure_error(
            &format!("a, b, {func}(b, 1) as c"),
            "requires an OVER clause",
        );
    }
}

#[test]
fn test_lead_lag_without_order_by() {
    // LAG/LEAD with OVER clause but without ORDER BY should raise an error
    for func in ["LAG", "LEAD"] {
        ensure_error(
            &format!("a, b, {func}(b) OVER (PARTITION BY a) as c"),
            "requires an ORDER BY",
        );
        ensure_error(
            &format!("a, b, {func}(b, 1) OVER (PARTITION BY a) as c"),
            "requires an ORDER BY",
        );
        // OVER clause with empty parentheses
        ensure_error(
            &format!("a, b, {func}(b) OVER () as c"),
            "requires an ORDER BY",
        );
    }
}
