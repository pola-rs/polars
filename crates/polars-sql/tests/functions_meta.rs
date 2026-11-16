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

#[test]
fn test_explain_does_not_run_other_statements() {
    let mut context = SQLContext::new();
    context.register("df", df! { "a" => [1, 2] }.unwrap().lazy());

    for sql in [
        "EXPLAIN DROP TABLE df",
        "EXPLAIN CREATE TABLE other AS SELECT * FROM df",
        "EXPLAIN INSERT INTO df VALUES (3)",
        "EXPLAIN WITH q AS (SELECT 3 AS a) INSERT INTO df SELECT a FROM q",
        "EXPLAIN TRUNCATE TABLE df",
    ] {
        let err = context.execute(sql).err().unwrap();
        assert!(err.to_string().contains("EXPLAIN supports queries only"));
    }

    assert_eq!(context.get_tables(), ["df"]);
    let df = context
        .execute("SELECT * FROM df")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df.height(), 2);
}
