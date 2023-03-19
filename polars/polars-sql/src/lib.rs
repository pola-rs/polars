mod context;
mod functions;
mod sql_expr;
mod table_functions;
pub use context::SQLContext;

#[cfg(test)]
mod test {
    use polars_core::prelude::*;
    use polars_lazy::dsl::lit;
    use polars_lazy::prelude::*;

    use super::*;

    fn create_sample_df() -> PolarsResult<DataFrame> {
        let a = Series::new("a", (1..10000i64).map(|i| i / 100).collect::<Vec<_>>());
        let b = Series::new("b", 1..10000i64);
        DataFrame::new(vec![a, b])
    }

    #[test]
    fn test_simple_select() -> PolarsResult<()> {
        let df = create_sample_df()?;
        let mut context = SQLContext::try_new()?;
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
        let mut context = SQLContext::try_new()?;
        context.register("df", df.clone().lazy());
        let df_sql = context
            .execute(r#"SELECT * FROM df WHERE (a > 3)"#)?
            .collect()?;
        let df_pl = df.lazy().filter(col("a").gt(lit(3))).collect()?;
        assert_eq!(df_sql, df_pl);
        Ok(())
    }

    #[test]
    fn test_groupby_simple() -> PolarsResult<()> {
        let df = create_sample_df()?;
        let mut context = SQLContext::try_new()?;
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
            .groupby(&[col("a")])
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
        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let sql = r#"
            SELECT 
                cast(a as FLOAT) as floats, 
                cast(a as INT) as ints, 
                cast(a as BIGINT) as bigints, 
                cast(a as STRING) as strings
            FROM df"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let df_pl = df
            .lazy()
            .select(&[
                col("a").cast(DataType::Float32).alias("floats"),
                col("a").cast(DataType::Int32).alias("ints"),
                col("a").cast(DataType::Int64).alias("bigints"),
                col("a").cast(DataType::Utf8).alias("strings"),
            ])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal(&df_pl));
    }

    #[test]
    fn test_literal_exprs() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal_missing(&df_pl));
    }

    #[test]
    fn test_prefixed_column_names() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal(&df_pl));
    }

    #[test]
    fn test_prefixed_column_names_2() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal(&df_pl));
    }
    #[test]
    fn test_binary_functions() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
            (col("a").cast(DataType::Utf8) + col("b").cast(DataType::Utf8)).alias("concat"),
        ]);
        let df_pl = df_pl.collect().unwrap();
        assert_eq!(df_sql, df_pl);
    }

    #[test]
    fn test_null_exprs() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal(&df_pl));
    }

    #[test]
    fn test_null_exprs_in_where() {
        let df = df! {
            "a" => &[Some(1), None, Some(3)],
            "b" => &[Some(1), Some(2), None]
        }
        .unwrap();

        let mut context = SQLContext::try_new().unwrap();
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

        assert!(df_sql.frame_equal_missing(&df_pl));
    }

    #[test]
    fn test_math_functions() {
        let df = df! {
            "a" => [1.0]
        }
        .unwrap();
        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let sql = r#"
            SELECT 
                a, 
                ABS(a) AS abs,
                ACOS(a) AS acos,
                ASIN(a) AS asin,
                ATAN(a) AS atan,
                CEIL(a) AS ceil,
                EXP(a) AS exp,
                FLOOR(a) AS floor,
                LN(a) AS ln,
                LOG2(a) AS log2,
                LOG10(a) AS log10,
                LOG(a, 5) AS log5,
                POW(a, 2) AS pow
            FROM df"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let df_pl = df
            .lazy()
            .select(&[
                col("a"),
                col("a").abs().alias("abs"),
                col("a").arccos().alias("acos"),
                col("a").arcsin().alias("asin"),
                col("a").arctan().alias("atan"),
                col("a").ceil().alias("ceil"),
                col("a").exp().alias("exp"),
                col("a").floor().alias("floor"),
                col("a").log(std::f64::consts::E).alias("ln"),
                col("a").log(2.0).alias("log2"),
                col("a").log(10.0).alias("log10"),
                col("a").log(5.0).alias("log5"),
                col("a").pow(2.0).alias("pow"),
            ])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal_missing(&df_pl));
    }

    #[test]
    fn test_iss_7440() {
        let df = df! {
            "a" => [2.0, -2.5]
        }
        .unwrap()
        .lazy();
        let sql = r#"SELECT a, FLOOR(a) AS floor, CEIL(a) AS ceil FROM df"#;
        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone());

        let df_sql = context.execute(sql).unwrap().collect().unwrap();

        let df_pl = df
            .select(&[
                col("a"),
                col("a").floor().alias("floor"),
                col("a").ceil().alias("ceil"),
            ])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal_missing(&df_pl));
    }
    #[test]
    fn test_string_functions() {
        let df = df! {
            "a" => &["foo", "xxxbarxxx", "---bazyyy"]
        }
        .unwrap();
        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let sql = r#"
            SELECT 
                a, 
                lower('LITERAL') as lower_literal,
                lower(a) as lower_a,
                lower("a") as lower_a2,
                lower(df.a) as lower_a_df,
                lower("df".a) as lower_a_df2,
                lower("df"."a") as lower_a_df3,
                upper(a) as upper_a,
                upper(df.a) as upper_a_df,
                upper("df".a) as upper_a_df2,
                upper("df"."a") as upper_a_df3,
                trim(both 'x' from a) as trim_a,
                trim(leading 'x' from a) as trim_a_leading,
                trim(trailing 'x' from a) as trim_a_trailing,
                ltrim(a) as ltrim_a,
                rtrim(a) as rtrim_a,
                ltrim(a, '-') as ltrim_a_dash,
                rtrim(a, '-') as rtrim_a_dash,
                ltrim(a, 'xyz') as ltrim_a_xyz,
                rtrim(a, 'xyz') as rtrim_a_xyz
            FROM df"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let df_pl = df
            .lazy()
            .select(&[
                col("a"),
                lit("LITERAL").str().to_lowercase().alias("lower_literal"),
                col("a").str().to_lowercase().alias("lower_a"),
                col("a").str().to_lowercase().alias("lower_a2"),
                col("a").str().to_lowercase().alias("lower_a_df"),
                col("a").str().to_lowercase().alias("lower_a_df2"),
                col("a").str().to_lowercase().alias("lower_a_df3"),
                col("a").str().to_uppercase().alias("upper_a"),
                col("a").str().to_uppercase().alias("upper_a_df"),
                col("a").str().to_uppercase().alias("upper_a_df2"),
                col("a").str().to_uppercase().alias("upper_a_df3"),
                col("a").str().strip(Some("x".into())).alias("trim_a"),
                col("a")
                    .str()
                    .lstrip(Some("x".into()))
                    .alias("trim_a_leading"),
                col("a")
                    .str()
                    .rstrip(Some("x".into()))
                    .alias("trim_a_trailing"),
                col("a").str().lstrip(None).alias("ltrim_a"),
                col("a").str().rstrip(None).alias("rtrim_a"),
                col("a")
                    .str()
                    .lstrip(Some("-".into()))
                    .alias("ltrim_a_dash"),
                col("a")
                    .str()
                    .rstrip(Some("-".into()))
                    .alias("rtrim_a_dash"),
                col("a")
                    .str()
                    .lstrip(Some("xyz".into()))
                    .alias("ltrim_a_xyz"),
                col("a")
                    .str()
                    .rstrip(Some("xyz".into()))
                    .alias("rtrim_a_xyz"),
            ])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal_missing(&df_pl));
    }
    #[test]
    fn test_agg_functions() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal(&df_pl));
    }
    #[test]
    fn create_table() {
        let df = create_sample_df().unwrap();
        let mut context = SQLContext::try_new().unwrap();
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
        assert!(df_sql.frame_equal(&create_tbl_res));
        let df_2 = context
            .execute(r#"SELECT a FROM df2"#)
            .unwrap()
            .collect()
            .unwrap();
        let expected = df.lazy().select(&[col("a")]).collect().unwrap();

        assert!(df_2.frame_equal(&expected));
    }
    #[test]
    fn test_unary_minus_0() {
        let df = df! {
            "value" => [-5, -3, 0, 3, 5],
        }
        .unwrap();

        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let sql = r#"SELECT * FROM df WHERE value < -1"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let df_pl = df
            .lazy()
            .filter(col("value").lt(lit(-1)))
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal(&df_pl));
    }
    #[test]
    fn test_unary_minus_1() {
        let df = df! {
            "value" => [-5, -3, 0, 3, 5],
        }
        .unwrap();

        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let sql = r#"SELECT * FROM df WHERE -value < 1"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let neg_value = lit(0) - col("value");
        let df_pl = df.lazy().filter(neg_value.lt(lit(1))).collect().unwrap();
        assert!(df_sql.frame_equal(&df_pl));
    }

    fn assert_sql_to_polars(df: &DataFrame, sql: &str, f: impl FnOnce(LazyFrame) -> LazyFrame) {
        let mut context = SQLContext::try_new().unwrap();
        context.register("df", df.clone().lazy());
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let df_pl = f(df.clone().lazy()).collect().unwrap();
        assert!(df_sql.frame_equal(&df_pl));
    }

    #[test]
    fn test_arr_agg() {
        let df = create_sample_df().unwrap();
        let exprs = vec![
            (
                "SELECT ARRAY_AGG(a) AS a FROM df",
                vec![col("a").list().alias("a")],
            ),
            (
                "SELECT ARRAY_AGG(a) AS a, ARRAY_AGG(b) as b FROM df",
                vec![col("a").list().alias("a"), col("b").list().alias("b")],
            ),
            (
                "SELECT ARRAY_AGG(a ORDER BY a) AS a FROM df",
                vec![col("a")
                    .sort_by(vec![col("a")], vec![false])
                    .list()
                    .alias("a")],
            ),
            (
                "SELECT ARRAY_AGG(a) AS a FROM df",
                vec![col("a").list().alias("a")],
            ),
            (
                "SELECT unnest(ARRAY_AGG(DISTINCT a)) FROM df",
                vec![col("a").unique_stable().list().explode().alias("a")],
            ),
            (
                "SELECT ARRAY_AGG(a ORDER BY b LIMIT 2) FROM df",
                vec![col("a")
                    .sort_by(vec![col("b")], vec![false])
                    .head(Some(2))
                    .list()],
            ),
        ];

        for (sql, expr) in exprs {
            assert_sql_to_polars(&df, sql, |df| df.select(&expr));
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn read_csv_tbl_func() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let create_tbl_res = df! {
            "Response" => ["Create Table"]
        }
        .unwrap();
        assert!(df_sql.frame_equal(&create_tbl_res));
        let df_2 = context
            .execute(r#"SELECT * FROM foods1"#)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(df_2.height(), 27);
        assert_eq!(df_2.width(), 4);
    }
    #[test]
    #[cfg(feature = "csv")]
    fn read_csv_tbl_func_inline() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            SELECT foods1.category
            FROM read_csv('../../examples/datasets/foods1.csv') as foods1"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();

        let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
            .finish()
            .unwrap()
            .select(&[col("category")])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal(&expected));
    }
    #[test]
    #[cfg(feature = "csv")]
    fn read_csv_tbl_func_inline_2() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            SELECT category
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();

        let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
            .finish()
            .unwrap()
            .select(&[col("category")])
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal(&expected));
    }

    #[test]
    #[cfg(feature = "parquet")]
    fn read_parquet_tbl() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_parquet('../../examples/datasets/foods1.parquet')"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let create_tbl_res = df! {
            "Response" => ["Create Table"]
        }
        .unwrap();
        assert!(df_sql.frame_equal(&create_tbl_res));
        let df_2 = context
            .execute(r#"SELECT * FROM foods1"#)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(df_2.height(), 27);
        assert_eq!(df_2.width(), 4);
    }
    #[test]
    #[cfg(feature = "ipc")]
    fn read_ipc_tbl() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_ipc('../../examples/datasets/foods1.ipc')"#;
        let df_sql = context.execute(sql).unwrap().collect().unwrap();
        let create_tbl_res = df! {
            "Response" => ["Create Table"]
        }
        .unwrap();
        assert!(df_sql.frame_equal(&create_tbl_res));
        let df_2 = context
            .execute(r#"SELECT * FROM foods1"#)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(df_2.height(), 27);
        assert_eq!(df_2.width(), 4);
    }

    #[test]
    #[cfg(feature = "csv")]
    fn iss_7436() {
        let mut context = SQLContext::try_new().unwrap();
        let sql = r#"
            CREATE TABLE foods AS
            SELECT *
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
        context.execute(sql).unwrap().collect().unwrap();
        let df_sql = context
            .execute(
                r#"
            SELECT 
                "fats_g" AS fats,
                AVG(calories) OVER (PARTITION BY "category") AS avg_calories_by_category
            FROM foods
            LIMIT 5
            "#,
            )
            .unwrap()
            .collect()
            .unwrap();
        let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
            .finish()
            .unwrap()
            .select(&[
                col("fats_g").alias("fats"),
                col("calories")
                    .mean()
                    .over(vec![col("category")])
                    .alias("avg_calories_by_category"),
            ])
            .limit(5)
            .collect()
            .unwrap();
        assert!(df_sql.frame_equal(&expected));
    }

    #[test]
    #[cfg(feature = "csv")]
    fn iss_7437() -> PolarsResult<()> {
        let mut context = SQLContext::try_new()?;
        let sql = r#"
            CREATE TABLE foods AS
            SELECT *
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
        context.execute(sql)?.collect()?;

        let df_sql = context
            .execute(
                r#"
                SELECT "category" as category
                FROM foods
                GROUP BY "category"
        "#,
            )?
            .collect()?
            .sort(&["category"], vec![false])?;

        let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
            .finish()?
            .groupby(vec![col("category").alias("category")])
            .agg(vec![])
            .collect()?
            .sort(&["category"], vec![false])?;

        assert!(df_sql.frame_equal(&expected));
        Ok(())
    }
    #[test]
    #[cfg(feature = "ipc")]
    fn test_groupby_2() -> PolarsResult<()> {
        let mut context = SQLContext::try_new()?;
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
        let expected =
            LazyFrame::scan_ipc("../../examples/datasets/foods1.ipc", Default::default())?
                .select(&[col("*")])
                .groupby(vec![col("category")])
                .agg(vec![
                    col("category").count().alias("count"),
                    col("calories").max(),
                    col("fats_g").min(),
                ])
                .sort_by_exprs(
                    vec![col("count"), col("category")],
                    vec![false, true],
                    false,
                )
                .limit(2);
        let lp = expected.clone().describe_optimized_plan()?;
        let expected = expected.collect()?;
        assert!(df_sql.frame_equal(&expected));
        Ok(())
    }
}
