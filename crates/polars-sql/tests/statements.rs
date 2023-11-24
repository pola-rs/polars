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
fn tbl_alias() {
    let mut ctx = create_ctx();
    let sql = r#"
    SELECT
        tbl.a,
        tbl.b,
    FROM df as tbl
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
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
fn select_qualified_wildcard() {
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
    assert!(actual.equals(&expected));
}

#[test]
fn select_qualified_column() {
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
        "b" => ["m", "n"],
        "a" => [2,3],
        "c" => ["y", "z"]
    ]
    .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("test", df1.lazy());
    ctx.register("test2", df2.lazy());

    let sql = r#"
    SELECT test.b, test2.*
    FROM test
    INNER JOIN test2
    USING(a)
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    assert!(actual.equals(&expected));
}

#[test]
fn test_union_all() {
    let df1 = df![
        "a" => [1,2,3],
        "b" => ["l", "m", "n"]
    ]
    .unwrap();
    let df2 = df![
        "a" => [4,2,3],
        "b" => ["x", "y", "z"]
    ]
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("test", df1.clone().lazy());
    ctx.register("test2", df2.clone().lazy());

    let sql = r#"
    SELECT * FROM test
    UNION ALL (
        SELECT * FROM test2
    )
    "#;
    let expected = polars_lazy::dsl::concat(
        vec![df1.lazy(), df2.lazy()],
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .unwrap()
    .collect()
    .unwrap();

    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    assert!(actual.equals(&expected));
}

#[test]
fn test_drop_table() {
    let mut ctx = create_ctx();
    let sql = r#"
    DROP TABLE df
    "#;
    let actual = ctx.execute(sql);
    assert!(actual.is_ok());
    let res = ctx.execute("SELECT * FROM df");
    assert!(res.is_err());
}

#[test]
fn iss_9560_join_as() {
    let df1 = df! {"id"=> [1, 2, 3, 4], "ano"=> [2, 3, 4, 5]}.unwrap();
    let df2 = df! {"id"=> [1, 2, 3, 4], "ano"=> [2, 3, 4, 5]}.unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());
    let sql = r#"
        SELECT * FROM df1 AS t1 JOIN df2 AS t2 ON t1.id = t2.id
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "id" => [1, 2, 3, 4],
        "ano" => [2, 3, 4, 5],
        "ano_right" => [2, 3, 4, 5],
    }
    .unwrap();

    assert!(actual.equals(&expected));
}
