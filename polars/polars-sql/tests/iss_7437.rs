use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
#[cfg(feature = "csv")]
fn iss_7437() -> PolarsResult<()> {
    let mut context = SQLContext::new();
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
        .sort(["category"], vec![false])?;

    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()?
        .groupby(vec![col("category").alias("category")])
        .agg(vec![])
        .collect()?
        .sort(["category"], vec![false])?;

    assert!(df_sql.frame_equal(&expected));
    Ok(())
}
