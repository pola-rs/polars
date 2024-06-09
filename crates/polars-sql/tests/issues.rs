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
        .sort(["category"], SortMultipleOptions::default())?;

    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()?
        .group_by(vec![col("category").alias("category")])
        .agg(vec![])
        .collect()?
        .sort(["category"], Default::default())?;

    assert!(df_sql.equals(&expected));
    Ok(())
}

#[test]
#[cfg(feature = "csv")]
fn iss_7436() {
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
    assert!(df_sql.equals(&expected));
}

#[test]
fn iss_7440() {
    let df = df! {
        "a" => [2.0, -2.5]
    }
    .unwrap()
    .lazy();
    let sql = r#"SELECT a, FLOOR(a) AS floor, CEIL(a) AS ceil FROM df"#;
    let mut context = SQLContext::new();
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
    assert!(df_sql.equals_missing(&df_pl));
}

#[test]
#[cfg(feature = "csv")]
fn iss_8395() -> PolarsResult<()> {
    use polars_core::series::Series;

    let mut context = SQLContext::new();
    let sql = r#"
    with foods as (
        SELECT *
        FROM read_csv('../../examples/datasets/foods1.csv')
    )
    select * from foods where category IN ('vegetables', 'seafood')"#;
    let res = context.execute(sql)?;
    let df = res.collect()?;

    // assert that the df only contains [vegetables, seafood]
    let s = df.column("category")?.unique()?.sort(Default::default())?;
    let expected = Series::new("category", &["seafood", "vegetables"]);
    assert!(s.equals(&expected));
    Ok(())
}

#[test]
fn iss_8419() {
    let df = df! {
      "Year"=> [2018, 2018, 2019, 2019, 2020, 2020],
      "Country"=> ["US", "UK", "US", "UK", "US", "UK"],
      "Sales"=> [1000, 2000, 3000, 4000, 5000, 6000]
    }
    .unwrap()
    .lazy();
    let expected = df
        .clone()
        .select(&[
            col("Year"),
            col("Country"),
            col("Sales"),
            col("Sales")
                .sort(SortOptions::default().with_order_descending(true))
                .cum_sum(false)
                .alias("SalesCumulative"),
        ])
        .sort(["SalesCumulative"], Default::default())
        .collect()
        .unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df", df);

    let query = r#"
    SELECT
        Year,
        Country,
        Sales,
        SUM(Sales) OVER (ORDER BY Sales DESC) as SalesCumulative
    FROM
        df
    ORDER BY
        SalesCumulative
    "#;
    let df = ctx.execute(query).unwrap().collect().unwrap();

    assert!(df.equals(&expected))
}
