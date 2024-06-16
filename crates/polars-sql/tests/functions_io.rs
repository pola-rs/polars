#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_core::prelude::*;
#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_lazy::prelude::*;
#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_sql::*;

#[test]
#[cfg(feature = "csv")]
fn read_csv_tbl_func() {
    let mut context = SQLContext::new();
    let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["CREATE TABLE"]
    }
    .unwrap();
    assert!(df_sql.equals(&create_tbl_res));
    let df_2 = context
        .execute(r#"SELECT * FROM foods1"#)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df_2.height(), 27);
    assert_eq!(df_2.width(), 4);
}

#[test]
#[cfg(feature = "csv")]
fn read_csv_tbl_func_inline() {
    let mut context = SQLContext::new();
    let sql = r#"
            SELECT foods1.category
            FROM read_csv('../../examples/datasets/foods1.csv') as foods1"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();

    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()
        .unwrap()
        .select(&[col("category")])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&expected));
}

#[test]
#[cfg(feature = "csv")]
fn read_csv_tbl_func_inline_2() {
    let mut context = SQLContext::new();
    let sql = r#"
            SELECT category
            FROM read_csv('../../examples/datasets/foods1.csv')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();

    let expected = LazyCsvReader::new("../../examples/datasets/foods1.csv")
        .finish()
        .unwrap()
        .select(&[col("category")])
        .collect()
        .unwrap();
    assert!(df_sql.equals(&expected));
}

#[test]
#[cfg(feature = "parquet")]
fn read_parquet_tbl() {
    let mut context = SQLContext::new();
    let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_parquet('../../examples/datasets/foods1.parquet')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["CREATE TABLE"]
    }
    .unwrap();
    assert!(df_sql.equals(&create_tbl_res));
    let df_2 = context
        .execute(r#"SELECT * FROM foods1"#)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df_2.height(), 27);
    assert_eq!(df_2.width(), 4);
}

#[test]
#[cfg(feature = "ipc")]
fn read_ipc_tbl() {
    let mut context = SQLContext::new();
    let sql = r#"
            CREATE TABLE foods1 AS
            SELECT *
            FROM read_ipc('../../examples/datasets/foods1.ipc')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["CREATE TABLE"]
    }
    .unwrap();
    assert!(df_sql.equals(&create_tbl_res));
    let df_2 = context
        .execute(r#"SELECT * FROM foods1"#)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df_2.height(), 27);
    assert_eq!(df_2.width(), 4);
}
