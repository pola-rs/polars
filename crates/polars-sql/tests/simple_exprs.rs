use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn create_sample_df() -> PolarsResult<DataFrame> {
    let a = Series::new("a", (1..10000i64).map(|i| i / 100).collect::<Vec<_>>());
    let b = Series::new("b", 1..10000i64);
    DataFrame::new(vec![a, b])
}

fn assert_sql_to_polars(df: &DataFrame, sql: &str, f: impl FnOnce(LazyFrame) -> LazyFrame) {
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = f(df.clone().lazy()).collect().unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_simple_select() -> PolarsResult<()> {
    let df = create_sample_df()?;
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let df_sql = context
        .execute(
            r#"
        SELECT a, b, a + b as c
        FROM df
        where a > 10 and a < 20
        LIMIT 100
    "#,
        )?
        .collect()?;
    let df_pl = df
        .lazy()
        .filter(col("a").gt(lit(10)).and(col("a").lt(lit(20))))
        .select(&[col("a"), col("b"), (col("a") + col("b")).alias("c")])
        .limit(100)
        .collect()?;
    assert_eq!(df_sql, df_pl);
    Ok(())
}

#[test]
fn test_nested_expr() -> PolarsResult<()> {
    let df = create_sample_df()?;
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let df_sql = context
        .execute(r#"SELECT * FROM df WHERE (a > 3)"#)?
        .collect()?;
    let df_pl = df.lazy().filter(col("a").gt(lit(3))).collect()?;
    assert_eq!(df_sql, df_pl);
    Ok(())
}
#[test]
fn test_group_by_simple() -> PolarsResult<()> {
    let df = create_sample_df()?;
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let df_sql = context
        .execute(
            r#"
        SELECT a, sum(b) as b , sum(a + b) as c, count(a) as total_count
        FROM df
        GROUP BY a
        LIMIT 100
    "#,
        )?
        .sort(
            "a",
            SortOptions {
                descending: false,
                nulls_last: false,
                ..Default::default()
            },
        )
        .collect()?;
    let df_pl = df
        .lazy()
        .group_by(&[col("a")])
        .agg(&[
            col("b").sum().alias("b"),
            (col("a") + col("b")).sum().alias("c"),
            col("a").count().alias("total_count"),
        ])
        .limit(100)
        .sort(
            "a",
            SortOptions {
                descending: false,
                nulls_last: false,
                ..Default::default()
            },
        )
        .collect()?;
    assert_eq!(df_sql, df_pl);
    Ok(())
}

#[test]
fn test_cast_exprs() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            cast(a as FLOAT) as floats,
            cast(a as INT) as ints,
            cast(a as BIGINT) as bigints,
            cast(a as STRING) as strings,
            cast(a as BLOB) as binary
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            col("a").cast(DataType::Float32).alias("floats"),
            col("a").cast(DataType::Int32).alias("ints"),
            col("a").cast(DataType::Int64).alias("bigints"),
            col("a").cast(DataType::String).alias("strings"),
            col("a").cast(DataType::Binary).alias("binary"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_literal_exprs() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            1 as int_lit,
            1.0 as float_lit,
            'foo' as string_lit,
            true as bool_lit,
            null as null_lit
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            lit(1i64).alias("int_lit"),
            lit(1.0).alias("float_lit"),
            lit("foo").alias("string_lit"),
            lit(true).alias("bool_lit"),
            lit(NULL).alias("null_lit"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals_missing(&df_pl));
}

#[test]
fn test_prefixed_column_names() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            df.a as a,
            df.b as b
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[col("a").alias("a"), col("b").alias("b")])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_prefixed_column_names_2() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            "df"."a" as a,
            "df"."b" as b
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[col("a").alias("a"), col("b").alias("b")])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_null_exprs() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            a,
            b,
            a is null as isnull_a,
            b is null as isnull_b,
            a is not null as isnotnull_a,
            b is not null as isnotnull_b
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            col("a"),
            col("b"),
            col("a").is_null().alias("isnull_a"),
            col("b").is_null().alias("isnull_b"),
            col("a").is_not_null().alias("isnotnull_a"),
            col("b").is_not_null().alias("isnotnull_b"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_null_exprs_in_where() {
    let df = df! {
        "a" => &[Some(1), None, Some(3)],
        "b" => &[Some(1), Some(2), None]
    }
    .unwrap();

    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            a,
            b
        FROM df
        WHERE a is null and b is not null"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .filter(col("a").is_null().and(col("b").is_not_null()))
        .collect()
        .unwrap();

    assert!(df_sql.equals_missing(&df_pl));
}

#[test]
fn binary_functions() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            a,
            b,
            a + b as add,
            a - b as sub,
            a * b as mul,
            a / b as div,
            a % b as rem,
            a <> b as neq,
            a = b as eq,
            a > b as gt,
            a < b as lt,
            a >= b as gte,
            a <= b as lte,
            a and b as and,
            a or b as or,
            a xor b as xor,
            a || b as concat
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df.lazy().select(&[
        col("a"),
        col("b"),
        (col("a") + col("b")).alias("add"),
        (col("a") - col("b")).alias("sub"),
        (col("a") * col("b")).alias("mul"),
        (col("a") / col("b")).alias("div"),
        (col("a") % col("b")).alias("rem"),
        col("a").eq(col("b")).not().alias("neq"),
        col("a").eq(col("b")).alias("eq"),
        col("a").gt(col("b")).alias("gt"),
        col("a").lt(col("b")).alias("lt"),
        col("a").gt_eq(col("b")).alias("gte"),
        col("a").lt_eq(col("b")).alias("lte"),
        col("a").and(col("b")).alias("and"),
        col("a").or(col("b")).alias("or"),
        col("a").xor(col("b")).alias("xor"),
        (col("a").cast(DataType::String) + col("b").cast(DataType::String)).alias("concat"),
    ]);
    let df_pl = df_pl.collect().unwrap();
    assert_eq!(df_sql, df_pl);
}

#[test]
#[ignore = "TODO: non deterministic"]
fn test_agg_functions() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            sum(a) as sum_a,
            first(a) as first_a,
            last(a) as last_a,
            avg(a) as avg_a,
            max(a) as max_a,
            min(a) as min_a,
            atan(a) as atan_a,
            stddev(a) as stddev_a,
            variance(a) as variance_a,
            count(a) as count_a,
            count(distinct a) as count_distinct_a,
            count(*) as count_all
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            col("a").sum().alias("sum_a"),
            col("a").first().alias("first_a"),
            col("a").last().alias("last_a"),
            col("a").mean().alias("avg_a"),
            col("a").max().alias("max_a"),
            col("a").min().alias("min_a"),
            col("a").arctan().alias("atan_a"),
            col("a").std(1).alias("stddev_a"),
            col("a").var(1).alias("variance_a"),
            col("a").count().alias("count_a"),
            col("a").n_unique().alias("count_distinct_a"),
            lit(1i32).count().alias("count_all"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn create_table() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        CREATE TABLE df2 AS
        SELECT a
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["Create Table"]
    }
    .unwrap();
    assert!(df_sql.equals(&create_tbl_res));
    let df_2 = context
        .execute(r#"SELECT a FROM df2"#)
        .unwrap()
        .collect()
        .unwrap();
    let expected = df.lazy().select(&[col("a")]).collect().unwrap();

    assert!(df_2.equals(&expected));
}

#[test]
fn test_unary_minus_0() {
    let df = df! {
        "value" => [-5, -3, 0, 3, 5],
    }
    .unwrap();

    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"SELECT * FROM df WHERE value < -1"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .filter(col("value").lt(lit(-1)))
        .collect()
        .unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_unary_minus_1() {
    let df = df! {
        "value" => [-5, -3, 0, 3, 5],
    }
    .unwrap();

    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"SELECT * FROM df WHERE -value < 1"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let neg_value = lit(0) - col("value");
    let df_pl = df.lazy().filter(neg_value.lt(lit(1))).collect().unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_arr_agg() {
    let df = create_sample_df().unwrap();
    let exprs = vec![
        (
            "SELECT ARRAY_AGG(a) AS a FROM df",
            vec![col("a").implode().alias("a")],
        ),
        (
            "SELECT ARRAY_AGG(a) AS a, ARRAY_AGG(b) as b FROM df",
            vec![col("a").implode().alias("a"), col("b").implode().alias("b")],
        ),
        (
            "SELECT ARRAY_AGG(a ORDER BY a) AS a FROM df",
            vec![col("a")
                .sort_by(vec![col("a")], vec![false])
                .implode()
                .alias("a")],
        ),
        (
            "SELECT ARRAY_AGG(a) AS a FROM df",
            vec![col("a").implode().alias("a")],
        ),
        (
            "SELECT unnest(ARRAY_AGG(DISTINCT a)) FROM df",
            vec![col("a").unique_stable().implode().explode().alias("a")],
        ),
        (
            "SELECT ARRAY_AGG(a ORDER BY b LIMIT 2) FROM df",
            vec![col("a")
                .sort_by(vec![col("b")], vec![false])
                .head(Some(2))
                .implode()],
        ),
    ];

    for (sql, expr) in exprs {
        assert_sql_to_polars(&df, sql, |df| df.select(&expr));
    }
}

#[test]
fn test_ctes() -> PolarsResult<()> {
    let df = create_sample_df()?;

    let mut context = SQLContext::new();
    context.register("df", df.lazy());

    let sql = r#"
    with df0 as (
        SELECT * FROM df
    )
    select * from df0 "#;
    assert!(context.execute(sql).is_ok());

    let sql = r#"select * from df0"#;
    assert!(context.execute(sql).is_err());

    Ok(())
}

#[test]
#[cfg(feature = "ipc")]
fn test_group_by_2() -> PolarsResult<()> {
    let mut context = SQLContext::new();
    let sql = r#"
    CREATE TABLE foods AS
    SELECT *
    FROM read_ipc('../../examples/datasets/foods1.ipc')"#;

    context.execute(sql)?.collect()?;
    let sql = r#"
    SELECT
        category,
        count(category) as count,
        max(calories),
        min(fats_g)
    FROM foods
    GROUP BY category
    ORDER BY count, category DESC
    LIMIT 2"#;

    let df_sql = context.execute(sql)?;
    let df_sql = df_sql.collect()?;
    let expected = LazyFrame::scan_ipc("../../examples/datasets/foods1.ipc", Default::default())?
        .select(&[col("*")])
        .group_by(vec![col("category")])
        .agg(vec![
            col("category").count().alias("count"),
            col("calories").max(),
            col("fats_g").min(),
        ])
        .sort_by_exprs(
            vec![col("count"), col("category")],
            vec![false, true],
            false,
            false,
        )
        .limit(2);
    let expected = expected.collect()?;
    assert!(df_sql.equals(&expected));
    Ok(())
}

#[test]
fn test_case_expr() {
    let df = create_sample_df().unwrap().head(Some(10));
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            CASE
                WHEN (a > 5 AND a < 8) THEN 'gt_5 and lt_8'
                WHEN a <= 5 THEN 'lteq_5'
                ELSE 'no match'
            END AS sign
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let case_expr = when(col("a").gt(lit(5)).and(col("a").lt(lit(8))))
        .then(lit("gt_5 and lt_8"))
        .when(col("a").lt_eq(lit(5)))
        .then(lit("lteq_5"))
        .otherwise(lit("no match"))
        .alias("sign");
    let df_pl = df.lazy().select(&[case_expr]).collect().unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_case_expr_with_expression() {
    let df = create_sample_df().unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            CASE b%2
                WHEN 0 THEN 'even'
                WHEN 1 THEN 'odd'
                ELSE 'No?'
            END AS parity
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let case_expr = when((col("b") % lit(2)).eq(lit(0)))
        .then(lit("even"))
        .when((col("b") % lit(2)).eq(lit(1)))
        .then(lit("odd"))
        .otherwise(lit("No?"))
        .alias("parity");
    let df_pl = df.lazy().select(&[case_expr]).collect().unwrap();
    assert!(df_sql.equals(&df_pl));
}

#[test]
fn test_sql_expr() {
    let df = create_sample_df().unwrap();
    let expr = sql_expr("MIN(a)").unwrap();
    let actual = df.clone().lazy().select(&[expr]).collect().unwrap();
    let expected = df.lazy().select(&[col("a").min()]).collect().unwrap();
    assert!(actual.equals(&expected));
}

#[test]
fn test_iss_9471() {
    let sql = r#"
    SELECT
        ABS(a,a,a,a,1,2,3,XYZRandomLetters,"XYZRandomLetters") as "abs",
    FROM df"#;
    let df = df! {
        "a" => [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    }
    .unwrap()
    .lazy();
    let mut context = SQLContext::new();
    context.register("df", df);
    let res = context.execute(sql);
    assert!(res.is_err())
}
