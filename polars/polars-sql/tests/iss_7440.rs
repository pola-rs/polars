use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

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
    assert!(df_sql.frame_equal_missing(&df_pl));
}
