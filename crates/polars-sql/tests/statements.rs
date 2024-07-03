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
        "id:t2" => [1, 2, 3, 4],
        "ano:t2" => [2, 3, 4, 5],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

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
        "b" => [0, 3, 4, 5, 7],
        "c" => [1, 3, 4, 5, 7]
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
        INNER JOIN df2 ON df1.a = df2.a AND df1.b = df2.b
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [2, 3],
        "b" => [3, 4],
        "a:df2" => [2, 3],
        "b:df2" => [3, 4],
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
        "c" => [7, 5, 3, 5, 7],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("lf1", df1.lazy());
    ctx.register("lf2", df2.lazy());

    let sql = r#"
        SELECT lf1.a, lf2.b, lf2.c
        FROM lf1 INNER JOIN lf2
          -- note: uses "lf1.a" for *both* constraint arms
          ON lf1.a = lf2.b AND lf1.a = lf2.c
        ORDER BY a
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    let expected = df! {
        "a" => [3, 5],
        "b" => [3, 5],
        "c" => [3, 5],
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
        SELECT df3.* FROM df1
          INNER JOIN df2
            ON df1.a = df2.a AND df1.b = df2.b
          INNER JOIN df3
            ON df3.a = df1.a AND df3.b = df1.b
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

    for cols in [
        "df1.*",
        "df2.*",
        "df1.a, df1.b, df2.c, df2.d",
        "df2.a, df2.b, df1.c, df1.d",
    ] {
        let sql = format!(
            r#"
            SELECT {} FROM df1
                INNER JOIN df2 ON
                    df1.a = df2.a AND
                    df1.b = df2.b AND
                    df1.c = df2.c AND
                    df1.d = df2.d
         "#,
            cols
        );
        let actual = ctx.execute(sql.as_str()).unwrap().collect().unwrap();
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
}

#[test]
fn test_resolve_join_column_select_13618() {
    let df1 = df! {
        "A" => [1, 2, 3, 4, 5],
        "B" => [5, 4, 3, 2, 1],
        "fruits" => ["banana", "banana", "apple", "apple", "banana"],
        "cars" => ["beetle", "audi", "beetle", "beetle", "beetle"],
    }
    .unwrap();
    let df2 = df1.clone();

    let mut ctx = SQLContext::new();
    ctx.register("tbl", df1.lazy());
    ctx.register("other", df2.lazy());

    let join_types = vec!["LEFT", "INNER", "FULL OUTER", ""];
    for join_type in join_types {
        let sql = format!(
            r#"
            SELECT tbl.A, other.B, tbl.fruits, other.cars
            FROM tbl
            {} JOIN other ON tbl.A = other.B
            ORDER BY tbl.A ASC
            "#,
            join_type
        );
        let actual = ctx.execute(sql.as_str()).unwrap().collect().unwrap();
        let expected = df! {
            "A" => [1, 2, 3, 4, 5],
            "B" => [1, 2, 3, 4, 5],
            "fruits" => ["banana", "banana", "apple", "apple", "banana"],
            "cars" => ["beetle", "beetle", "beetle", "audi", "beetle"],
        }
        .unwrap();

        assert!(
            actual.equals(&expected),
            "({} JOIN) expected = {:?}\nactual={:?}",
            join_type,
            expected,
            actual
        );
    }
}

#[test]
fn test_compound_join_and_select_exclude_rename_replace() {
    let df1 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 2, 3, 4, 5],
        "c" => [0, 3, 4, 5, 6],
        "d" => [0, 3, 4, 5, 6],
        "e" => ["a", "b", "c", "d", "?"],
    }
    .unwrap();
    let df2 = df! {
        "a" => [1, 2, 3, 4, 5],
        "b" => [1, 3, 3, 5, 6],
        "c" => [0, 3, 4, 5, 6],
        "d" => [0, 3, 4, 5, 6],
        "e" => ["w", "x", "y", "z", "!"],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let sql = r#"
        SELECT * RENAME ("ee" AS "e")
        FROM (
          SELECT df1.* EXCLUDE "e", df2.e AS "ee"
          FROM df1
            INNER JOIN df2 ON df1.a = df2.a AND
              ((df1.b = df2.b AND df1.c = df2.c) AND df1.d = df2.d)
        ) tbl
     "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    let expected = df! {
        "a" => [1, 3],
        "b" => [1, 3],
        "c" => [0, 4],
        "d" => [0, 4],
        "e" => ["w", "y"],
    }
    .unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );

    let sql = r#"
        SELECT * REPLACE ("ee" || "ee" AS "ee")
        FROM (
          SELECT * EXCLUDE ("e", "e:df2"), df1.e AS "ee"
          FROM df1
            INNER JOIN df2 ON df1.a = df2.a AND
              ((df1.b = df2.b AND df1.c = df2.c) AND df1.d = df2.d)
        ) tbl
     "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1, 3],
        "b" => [1, 3],
        "c" => [0, 4],
        "d" => [0, 4],
        "a:df2" => [1, 3],
        "b:df2" => [1, 3],
        "c:df2" => [0, 4],
        "d:df2" => [0, 4],
        "ee" => ["aa", "cc"],
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
fn test_join_on_different_keys() {
    let df1 = df! {"x" => [-1, 0, 1, 2, 3, 4]}.unwrap();
    let df2 = df! {"y" => [0, 1, -2, 3, 5, 6]}.unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    // join on x = y
    let sql = r#"
        SELECT df2.*
        FROM df1
        INNER JOIN df2 ON df1.x = df2.y
        ORDER BY y
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();
    let expected = df! {"y" => [0, 1, 3]}.unwrap();
    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
fn test_join_multi_consecutive() {
    let df1 = df! { "a" => [1, 2, 3], "b" => [4, 8, 6] }.unwrap();
    let df2 = df! { "a" => [3, 2, 1], "b" => [6, 5, 4], "c" => ["x", "y", "z"] }.unwrap();
    let df3 = df! { "c" => ["w", "y", "z"], "d" => [10.5, -50.0, 25.5] }.unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("tbl_a", df1.lazy());
    ctx.register("tbl_b", df2.lazy());
    ctx.register("tbl_c", df3.lazy());

    let sql = r#"
        SELECT tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d FROM tbl_a
        INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
        INNER JOIN tbl_c ON tbl_a.c = tbl_c.c
        ORDER BY a DESC
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    let expected = df! {
        "a" => [1],
        "b" => [4],
        "c" => ["z"],
        "d" => [25.5],
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
fn test_join_utf8() {
    // (色) color and (野菜) vegetable
    let df1 = df! {
        "色" => ["赤", "緑", "黄色"],
        "野菜" => ["トマト", "ケール", "コーン"],
    }
    .unwrap();

    // (色) color and (動物) animal
    let df2 = df! {
        "色" => ["黄色", "緑", "赤"],
        "動物" => ["ゴシキヒワ", "蛙", "レッサーパンダ"],
    }
    .unwrap();

    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let expected = df! {
        "色" => ["緑", "赤", "黄色"],  // green, red, yellow
        "野菜" => ["ケール", "トマト", "コーン"],  // kale, tomato, corn
        "動物" => ["蛙", "レッサーパンダ", "ゴシキヒワ"],  // frog, red panda, goldfinch
    }
    .unwrap();

    let sql = r#"
        SELECT df1.*, df2.動物
        FROM df1
        INNER JOIN df2 ON df1.色 = df2.色
        ORDER BY 色
    "#;
    let actual = ctx.execute(sql).unwrap().collect().unwrap();

    assert!(
        actual.equals(&expected),
        "expected = {:?}\nactual={:?}",
        expected,
        actual
    );
}

#[test]
fn test_table() {}

#[test]
#[should_panic]
fn test_compound_invalid_1() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 OUTER JOIN df2 ON a AND b";
    ctx.execute(sql).unwrap().collect().unwrap();
}

#[test]
#[should_panic]
fn test_compound_invalid_2() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 LEFT JOIN df2 ON df1.a = df2.a AND b = b";
    ctx.execute(sql).unwrap().collect().unwrap();
}

#[test]
#[should_panic]
fn test_compound_invalid_3() {
    let mut ctx = prepare_compound_join_context();
    let sql = "SELECT * FROM df1 INNER JOIN df2 ON df1.a = df2.a AND b";
    ctx.execute(sql).unwrap().collect().unwrap();
}
