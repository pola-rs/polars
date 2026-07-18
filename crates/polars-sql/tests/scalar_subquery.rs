use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_ctx() -> SQLContext {
    let df = df! {
        "a" => [1, 2, 3, 4, 5],
    }
    .unwrap()
    .lazy();
    let threshold = df! {
        "x" => [3],
    }
    .unwrap()
    .lazy();

    let ctx = SQLContext::new();
    ctx.register("df", df);
    ctx.register("threshold", threshold);
    ctx
}

fn assert_scalar_subquery(op: &str, expected_a: &[i32]) {
    let mut ctx = create_ctx();
    let query = format!("SELECT a FROM df WHERE a {op} (SELECT x FROM threshold) ORDER BY a");
    let actual = ctx.execute(&query).unwrap().collect().unwrap();

    let expected = df! { "a" => expected_a }.unwrap();
    assert!(
        expected.equals(&actual),
        "op={op}\nexpected={expected:?}\nactual={actual:?}"
    );
}

#[test]
fn test_scalar_subquery_eq() {
    assert_scalar_subquery("=", &[3]);
}

#[test]
fn test_scalar_subquery_not_eq() {
    assert_scalar_subquery("<>", &[1, 2, 4, 5]);
}

#[test]
fn test_scalar_subquery_gt() {
    assert_scalar_subquery(">", &[4, 5]);
}

#[test]
fn test_scalar_subquery_gt_eq() {
    assert_scalar_subquery(">=", &[3, 4, 5]);
}

#[test]
fn test_scalar_subquery_lt() {
    assert_scalar_subquery("<", &[1, 2]);
}

#[test]
fn test_scalar_subquery_lt_eq() {
    assert_scalar_subquery("<=", &[1, 2, 3]);
}

#[test]
fn test_scalar_subquery_reversed_operands() {
    // (subquery) on the left-hand side of the comparison.
    let mut ctx = create_ctx();
    let actual = ctx
        .execute("SELECT a FROM df WHERE (SELECT x FROM threshold) < a ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();

    let expected = df! { "a" => [4, 5] }.unwrap();
    assert!(expected.equals(&actual));
}

#[test]
fn test_scalar_subquery_unsupported_op() {
    let mut ctx = create_ctx();
    let res = ctx.execute("SELECT a FROM df WHERE a + (SELECT x FROM threshold) > 0");
    assert!(res.is_err());
}
