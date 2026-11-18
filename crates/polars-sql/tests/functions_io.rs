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

#[test]
#[cfg(feature = "csv")]
fn copy_to_file() {
    let dir = std::env::temp_dir().join(format!("polars_sql_copy_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("foods.csv");
    let target = path.to_str().unwrap().replace('\\', "/");

    let mut context = SQLContext::new();
    let query = "SELECT category, calories FROM '../../examples/datasets/foods1.csv' \
                 WHERE calories > 100";
    let expected = context.execute(query).unwrap().collect().unwrap();

    // The plan writes the file once it is collected.
    let plan = context
        .execute(&format!("COPY ({query}) TO '{target}'"))
        .unwrap();
    assert!(!path.exists());
    assert_eq!(plan.collect().unwrap().shape(), (0, 0));
    let written = context
        .execute(&format!("SELECT * FROM '{target}'"))
        .unwrap()
        .collect()
        .unwrap();
    assert!(written.equals(&expected));

    // A table with a column list, replacing the file; `FORMAT` and the CSV options.
    let t = df! { "a" => [1, 2], "b" => ["x", "y"] }.unwrap();
    context.register("t", t.lazy());
    let other = dir.join("foods.txt");
    for (sql, file, contents) in [
        ("COPY t (b, a) TO '{}'", &path, "b,a\nx,1\ny,2\n"),
        (
            "COPY t TO '{}' (FORMAT csv, HEADER false, DELIMITER '|')",
            &other,
            "1|x\n2|y\n",
        ),
    ] {
        let sql = sql.replace("{}", &file.to_str().unwrap().replace('\\', "/"));
        context.execute(&sql).unwrap().collect().unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), contents);
    }

    // Writing files can be turned off.
    let mut context = SQLContext::new().with_file_writes(false);
    let err = context
        .execute(&format!("COPY (SELECT 1 AS a) TO '{target}'"))
        .err()
        .unwrap();
    assert!(err.to_string().contains("writing files is disabled"));

    std::fs::remove_dir_all(&dir).unwrap();
}
