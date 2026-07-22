use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn df() -> DataFrame {
    df! {
        "a" => [1i32, 2, 3],
        "b" => [10i32, 20, 5],
        "c" => [100i32, 200, 8],
    }
    .unwrap()
}

fn ctx() -> SQLContext {
    let ctx = SQLContext::new();
    ctx.register("t1", df().lazy());
    ctx
}

#[test]
fn test_binary_expr_collides_with_bare_column() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a-b, a FROM t1 ORDER BY a-b")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "a:1"]
    );
    let a_minus_b: Vec<i32> = out
        .column("a")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let a: Vec<i32> = out
        .column("a:1")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(a_minus_b, vec![-18, -9, -2]);
    assert_eq!(a, vec![2, 1, 3]);
}

#[test]
fn test_function_call_collides_with_bare_column() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a, abs(a) FROM t1 ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "a:1"]
    );
    let col_a: Vec<i32> = out
        .column("a")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let col_abs_a: Vec<i32> = out
        .column("a:1")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(col_a, vec![1, 2, 3]);
    assert_eq!(col_abs_a, vec![1, 2, 3]);
}

#[test]
fn test_two_derived_expressions_collide() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a-b, a-c FROM t1 ORDER BY a-b")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "a:1"]
    );
}

#[test]
fn test_no_collision_output_names_are_unchanged() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a, b, a + b AS c2 FROM t1 ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c2"]
    );
}

#[test]
fn test_explicit_alias_takes_priority_over_derived_name() {
    // The unaliased "a-b" must yield to the explicit alias "a" declared later
    // in the SELECT list, even though it would otherwise claim that name first.
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a-b, 5 AS a FROM t1 ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a:1", "a"]
    );
    let a: Vec<i32> = out
        .column("a")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(a, vec![5, 5, 5]);
}

#[test]
fn test_explicit_duplicate_alias_still_errors() {
    let mut ctx = ctx();
    let res = ctx.execute("SELECT 1 AS x, 2 AS x FROM t1");
    let res = res.and_then(|lf| lf.collect());
    assert!(res.is_err());
}

#[test]
fn test_verbatim_duplicate_selection_still_errors() {
    let mut ctx1 = ctx();
    let res = ctx1.execute("SELECT a, a FROM t1");
    assert!(res.and_then(|lf| lf.collect()).is_err());

    let mut ctx2 = ctx();
    let res = ctx2.execute("SELECT *, a FROM t1");
    assert!(res.and_then(|lf| lf.collect()).is_err());
}

#[test]
fn test_wildcard_collides_with_derived_expression() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT *, a-b FROM t1 ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c", "a:1"]
    );
}

#[test]
fn test_collision_with_group_by() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT a, abs(a) FROM t1 GROUP BY a ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "a:1"]
    );
}

#[test]
fn test_collision_with_distinct() {
    let mut ctx = ctx();
    let out = ctx
        .execute("SELECT DISTINCT a-b, a FROM t1 ORDER BY a-b")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        out.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "a:1"]
    );
    assert_eq!(out.height(), 3);
}
