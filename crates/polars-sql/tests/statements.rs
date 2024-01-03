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

// Tests for https://github.com/pola-rs/polars/issues/11290 --------------

fn prepare_compound_join_context() -> SQLContext {
    let df1 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 3, 4, 4, 5],
    }
    .unwrap();
    let df2 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [0, 3, 4, 5, 6]
    }
    .unwrap();

    let df3 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [0, 3, 4, 5, 6],
        "c" => [0, 3, 4, 5, 6]
    }
    .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());
    ctx.register("df3", df3.lazy());
    ctx
}

#[test]
fn test_compound_join_basic() {
    let mut ctx = prepare_compound_join_context();
    let sql = r#"
        SELECT * FROM df1
        JOIN df2 ON df1.a = df2.a AND df1.b = df2.b
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [2, 3],
        "b" => [3, 4],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
fn test_compound_join_different_column_names() {
    let df1 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 2, 3, 4, 5],
    }
    .unwrap();
    let df2 = df! {
        "a" => [0, 2, 3, 4, 5],
        "b" => [1, 2, 3, 5, 6],
        "c" => [7, 8, 9, 10, 11],
    }
    .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let sql = r#"
        SELECT * FROM df1 JOIN df2 ON df1.a = df2.b AND df1.b = df2.a
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [2, 3],
        "b" => [2, 3],
        "c" => [8, 9],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
fn test_compound_join_three_tables() {
    let mut ctx = prepare_compound_join_context();
    let sql = r#"
        SELECT * FROM df1
            JOIN df2
                ON df1.a = df2.a AND df1.b = df2.b
            JOIN df3
                ON df1.a = df3.a AND df1.b = df3.b
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [2, 3],
        "b" => [3, 4],
        "c" => [3, 4],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
fn test_compound_join_nested_and() {
    let df1 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 2, 3, 4, 5],
        "c" => [0, 3, 4, 5, 6],
        "d" => [0, 3, 4, 5, 6],
    }
    .unwrap();
    let df2 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 3, 3, 5, 6],
        "c" => [0, 3, 4, 5, 6],
        "d" => [0, 3, 4, 5, 6]
    }
    .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let sql = r#"
        SELECT * FROM df1
            JOIN df2 ON
                df1.a = df2.a AND
                df1.b = df2.b AND
                df1.c = df2.c AND
                df1.d = df2.d
     "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1, 3],
        "b" => [1, 3],
        "c" => [0, 4],
        "d" => [0, 4],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
#[should_panic]
fn test_compound_invalid_1() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 JOIN df2 ON a AND b";
    ctx.execute(sql).unwrap().collect().unwrap();
}

#[test]
#[should_panic]
fn test_compound_invalid_2() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 JOIN df2 ON df1.a = df2.a AND b = b";
    ctx.execute(sql).unwrap().collect().unwrap();
}

#[test]
#[should_panic]
fn test_compound_invalid_3() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 JOIN df2 ON df1.a = df2.a AND b";
    ctx.execute(sql).unwrap().collect().unwrap();
}
