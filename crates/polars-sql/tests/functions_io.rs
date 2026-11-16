#[cfg(any(feature = "csv", feature = "ipc", feature = "parquet"))]
use polars_core::prelude::*;
#[cfg(feature = "csv")]
use polars_lazy::prelude::*;
#[cfg(any(feature = "csv", feature = "ipc", feature = "parquet"))]
use polars_sql::*;
#[cfg(feature = "csv")]
use polars_utils::pl_path::PlRefPath;

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
        "Response" => ["CREATE TABLE foods1"]
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

    let expected = LazyCsvReader::new(PlRefPath::new("../../examples/datasets/foods1.csv"))
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

    let expected = LazyCsvReader::new(PlRefPath::new("../../examples/datasets/foods1.csv"))
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
            FROM Read_Parquet('../../examples/datasets/foods1.parquet')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["CREATE TABLE foods1"]
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
            FROM READ_IPC('../../examples/datasets/foods1.ipc')"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let create_tbl_res = df! {
        "Response" => ["CREATE TABLE foods1"]
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
fn from_quoted_path() {
    fn run(context: &mut SQLContext, sql: &str) -> DataFrame {
        context.execute(sql).unwrap().collect().unwrap()
    }
    let mut context = SQLContext::new();
    let path = "../../examples/datasets/foods1.csv";

    // Read as `read_csv` reads it.
    let expected = run(&mut context, &format!("SELECT * FROM read_csv('{path}')"));
    let df = run(&mut context, &format!("SELECT * FROM '{path}'"));
    assert!(df.equals_missing(&expected));
    assert_eq!(df.shape(), (27, 4));

    // With an alias, in a join, and as a glob.
    let df = run(
        &mut context,
        &format!(
            "SELECT a.category, b.calories FROM '{path}' AS a \
             JOIN '../../examples/datasets/foods2.csv' AS b ON a.category = b.category \
             WHERE a.calories > 100 ORDER BY ALL"
        ),
    );
    let expected = run(
        &mut context,
        "SELECT a.category, b.calories FROM read_csv('../../examples/datasets/foods1.csv') AS a \
         JOIN read_csv('../../examples/datasets/foods2.csv') AS b ON a.category = b.category \
         WHERE a.calories > 100 ORDER BY ALL",
    );
    assert!(df.height() > 0);
    assert!(df.equals(&expected));
    let glob = "../../examples/datasets/foods*.csv";
    let expected = run(
        &mut context,
        &format!("SELECT COUNT(*) AS n FROM read_csv('{glob}')"),
    );
    let df = run(&mut context, &format!("SELECT COUNT(*) AS n FROM '{glob}'"));
    assert!(df.equals(&expected));

    // `DESCRIBE` gives the columns that `FROM` reads.
    let expected = run(&mut context, &format!("DESCRIBE SELECT * FROM '{path}'"));
    let df = run(&mut context, &format!("DESCRIBE '{path}'"));
    assert!(df.equals(&expected));
    assert_eq!(df.height(), 4);

    // Any other name in quotes is a table.
    for name in ["foods", "foods1.txt", "../../examples/datasets"] {
        let lf = context.execute(&format!("SELECT * FROM '{name}'"));
        let err = lf.and_then(|lf| lf.collect()).err().unwrap().to_string();
        assert!(
            err.contains(&format!("relation '{name}' was not found")),
            "{err}"
        );
        assert!(err.contains("read_csv"), "{err}");
    }

    // A file is read, not written to.
    let err = context.execute(&format!("DELETE FROM '{path}' WHERE calories > 0"));
    assert!(err.err().unwrap().to_string().contains("DELETE of a file"));

    // A table of that name comes first.
    context.register(path, df! { "a" => [1] }.unwrap().lazy());
    let df = run(&mut context, &format!("SELECT * FROM '{path}'"));
    assert_eq!(df.get_column_names(), ["a"]);
    let df = run(&mut context, &format!("DELETE FROM '{path}' WHERE a = 1"));
    assert_eq!(df.shape(), (0, 1));
}
