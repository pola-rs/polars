use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
fn test_describe() {
    let lf = df! {
      "year"=> [2018],
      "country"=> ["US"],
      "sales"=> [1000.0]
    }
    .unwrap()
    .lazy();
    let mut context = SQLContext::new();
    context.register("df", lf.clone());
    let sql = r#"EXPLAIN SELECT MAX(year) as year_max, MIN(country) AS country FROM df"#;
    let res = context.execute(sql).unwrap();
    let df = res.collect().unwrap();
    let lf = lf.select([
        col("year").max().alias("year_max"),
        col("country").min().alias("country"),
    ]);
    let expected = lf.describe_optimized_plan().unwrap();

    let expected = expected.split('\n').map(Some).collect::<Vec<_>>();
    let actual = df
        .column("Logical Plan")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}
