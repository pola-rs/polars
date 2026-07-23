use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn small_frames() -> (LazyFrame, LazyFrame) {
    let left = df! {
        "a" => [1, 2, 3],
    }
    .unwrap()
    .lazy();
    let right = df! {
        "b" => [10, 20],
    }
    .unwrap()
    .lazy();
    (left, right)
}

#[test]
fn inner_join_constant_true_on_is_cross_join() {
    let (left, right) = small_frames();
    let mut ctx = SQLContext::new();
    ctx.register("t1", left);
    ctx.register("t2", right);

    let sql = "SELECT * FROM t1 INNER JOIN t2 ON 1 IS NOT NULL";
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1, 1, 2, 2, 3, 3],
        "b" => [10, 20, 10, 20, 10, 20],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {expected:?}\nactual = {actual:?}"
    );
}

#[test]
fn inner_join_constant_false_on_is_empty() {
    let (left, right) = small_frames();
    let mut ctx = SQLContext::new();
    ctx.register("t1", left);
    ctx.register("t2", right);

    let sql = "SELECT * FROM t1 INNER JOIN t2 ON NULL IS NOT NULL";
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    assert_eq!(actual.height(), 0);
}

#[test]
fn left_outer_join_constant_true_on_is_cross_join() {
    let (left, right) = small_frames();
    let mut ctx = SQLContext::new();
    ctx.register("t1", left);
    ctx.register("t2", right);

    let sql = "SELECT * FROM t1 LEFT OUTER JOIN t2 ON 1 IS NOT NULL";
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1, 1, 2, 2, 3, 3],
        "b" => [10, 20, 10, 20, 10, 20],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {expected:?}\nactual = {actual:?}"
    );
}

#[test]
fn left_outer_join_constant_false_on_keeps_all_left_rows_with_null_right() {
    let (left, right) = small_frames();
    let mut ctx = SQLContext::new();
    ctx.register("t1", left);
    ctx.register("t2", right);

    let sql = "SELECT * FROM t1 LEFT OUTER JOIN t2 ON NULL IS NOT NULL";
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1, 2, 3],
        "b" => [None::<i32>, None, None],
    }
    .unwrap();

    assert!(
        actual.equals_missing(&expected),
        "expected = {expected:?}\nactual = {actual:?}"
    );
}
