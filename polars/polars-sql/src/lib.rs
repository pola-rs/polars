mod context;
mod sql_expr;

pub use context::SQLContext;

#[cfg(test)]
mod test {
    use polars_core::prelude::*;
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
