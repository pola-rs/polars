use polars_core::chunked_array::ops::SortMultipleOptions;
use polars_core::df;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
fn test_distinct_on() {
    let df = df! {
      "Name" => ["Bob", "Pete", "Pete", "Pete", "Martha", "Martha"],
      "Record Date" => [1, 1, 2, 4, 1, 3],
      "Score" => [8, 2, 9, 3, 2, 6]
    }
    .unwrap()
    .lazy();
    let mut ctx = SQLContext::new();

    ctx.register("df", df.clone());
    let sql = r#"
      SELECT DISTINCT ON ("Name")
          "Name",
          "Record Date",
          "Score"
      FROM
          df
      ORDER BY
          "Name",
          "Record Date" DESC;"#;
    let lf = ctx.execute(sql).unwrap();
    let actual = lf.collect().unwrap();
    let expected = df
        .sort_by_exprs(
            vec![col("Name"), col("Record Date")],
            SortMultipleOptions::default()
                .with_order_descending_multi([false, true])
                .with_maintain_order(true),
        )
        .group_by_stable(vec![col("Name")])
        .agg(vec![col("*").first()]);
    let expected = expected.collect().unwrap();
    assert!(actual.equals(&expected))
}
