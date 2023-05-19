use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_ctx() -> SQLContext {
    let a = Series::new("a", (1..10i64).map(|i| i / 100).collect::<Vec<_>>());
    let b = Series::new("b", 1..10i64);
    let df = DataFrame::new(vec![a, b]).unwrap().lazy();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);
    ctx
}

#[test]
fn trailing_commas_allowed() {
    let mut ctx = create_ctx();
    let sql = r#"
    SELECT
        a,
        b,
    FROM df
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
}

#[test]
fn select_exclude_single() {
    let mut ctx = create_ctx();
    let sql = r#"
    SELECT * EXCLUDE a FROM df
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
}

#[test]
fn select_exclude_multi() {
    let mut ctx = create_ctx();
    let sql = r#"
    SELECT * EXCLUDE (a) FROM df
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
}

#[test]
fn select_from_join() {
    let df1 = df![
        "a" => [1,2,3],
        "b" => ["l", "m", "n"]
    ]
    .unwrap();
    let df2 = df![
        "a" => [4,2,3],
        "c" => ["x", "y", "z"]
    ]
    .unwrap();

    let expected = df![
        "a" => [2,3],
        "b" => ["m", "n"]
    ]
    .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("test", df1.lazy());
    ctx.register("test2", df2.lazy());

    let sql = r#"
    SELECT test.* 
    FROM test 
    INNER JOIN test2 
    USING(a)
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    assert!(actual.frame_equal(&expected));
}
