use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

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
                .sort(true)
                .cum_sum(false)
                .alias("SalesCumulative"),
        ])
        .sort("SalesCumulative", SortOptions::default())
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
