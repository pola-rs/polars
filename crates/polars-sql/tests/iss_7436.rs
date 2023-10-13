#[test]
#[cfg(feature = "csv")]
fn iss_7436() {
    use polars_lazy::prelude::*;
    use polars_sql::*;

    let mut context = SQLContext::new();
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
