pub use context::SQLContext;

mod context;
mod sql_expr;

#[cfg(test)]
mod test {
    use super::*;
    use polars::prelude::*;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    #[test]
    fn test_expr() {
        let dialect = GenericDialect {};
        let sql =
            "SELECT a, (b::int + a)/a as c, count(b) + (-1.0) as b1 FROM t group by a limit 100;";
        let ast = Parser::parse_sql(&dialect, sql).unwrap();
        if !ast.is_empty() {
            println!("{:?}", ast);
        };
    }

    fn create_sample_df() -> Result<DataFrame> {
        let a = Series::new("a", (1..10000i64).map(|i| i / 100).collect::<Vec<_>>());
        let b = Series::new("b", 1..10000i64);
        DataFrame::new(vec![a, b])
    }

    #[test]
    fn test_simple_select() -> Result<()> {
        let df = create_sample_df()?;
        let mut context = SQLContext::new();
        context.register("df", &df);
        let df_sql = context
            .execute(
                r#"
            SELECT a, b, a + b as c
            FROM df
            LIMIT 100
        "#,
            )?
            .collect()?;
        let df_pl = df
            .lazy()
            .select(&[col("a"), col("b"), (col("a") + col("b")).alias("c")])
            .limit(100)
            .collect()?;
        assert_eq!(df_sql, df_pl);
        Ok(())
    }

    #[test]
    fn test_groupby_simple() -> Result<()> {
        let df = create_sample_df()?;
        let mut context = SQLContext::new();
        context.register("df", &df);
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
                },
            )
            .collect()?;
        assert_eq!(df_sql, df_pl);
        Ok(())
    }
}
