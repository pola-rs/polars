#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_core::prelude::*;
#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_lazy::prelude::*;
#[cfg(any(feature = "csv", feature = "ipc"))]
use polars_sql::*;
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

#[cfg(feature = "csv")]
mod read_csv_options {
    use super::*;

    /// Write `contents` to `name` in `dir` and return its path.
    fn write_file(dir: &tempfile::TempDir, name: &str, contents: &[u8]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn query(sql: &str) -> PolarsResult<DataFrame> {
        SQLContext::new().execute(sql)?.collect()
    }

    fn query_err(sql: &str) -> String {
        query(sql).unwrap_err().to_string()
    }

    fn dtypes(df: &DataFrame) -> Vec<DataType> {
        df.dtypes()
    }

    #[test]
    fn headerless_ragged_rows() {
        let dir = tempfile::tempdir().unwrap();
        // From the issue: without a header, short rows are padded with nulls.
        let path = write_file(&dir, "ragged.csv", b"1,a\n2,b,extra\n3\n");
        let err = query_err(&format!("SELECT * FROM read_csv('{path}')"));
        assert!(err.contains("found more fields than defined"), "{err}");

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', has_header => false)"
        ))
        .unwrap();
        let expected = df! {
            "column_0" => [1i64, 2, 3],
            "column_1" => [Some("a"), Some("b"), None],
            "column_2" => [None, Some("extra"), None],
        }
        .unwrap();
        assert!(df.equals_missing(&expected), "{df}");
    }

    #[test]
    fn wide_row_after_inference_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut contents: String = (1..=150).map(|i| format!("{i},b\n")).collect();
        contents.push_str("151,b,late\n");
        let path = write_file(&dir, "late.csv", contents.as_bytes());

        let err = query_err(&format!(
            "SELECT * FROM read_csv('{path}', has_header => false)"
        ));
        assert!(err.contains("found more fields than defined"), "{err}");

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', has_header => false, infer_schema_length => NULL)"
        ))
        .unwrap();
        assert_eq!(df.shape(), (151, 3));
        assert_eq!(df.column("column_2").unwrap().null_count(), 150);
    }

    #[test]
    fn operators_and_case_insensitive_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "ops.csv", b"1,2\n");
        for options in [
            "has_header => false",
            "has_header := false",
            "HAS_HEADER => false",
        ] {
            let df = query(&format!("SELECT * FROM read_csv('{path}', {options})")).unwrap();
            assert_eq!(df.shape(), (1, 2), "{options}");
        }
    }

    #[test]
    fn layout_options() {
        let dir = tempfile::tempdir().unwrap();
        let semicolons = write_file(&dir, "semicolons.csv", b"a;b\n1;\"x;y\"\n");
        let df = query(&format!(
            "SELECT b FROM read_csv('{semicolons}', separator => ';')"
        ))
        .unwrap();
        assert_eq!(df.column("b").unwrap().str().unwrap().get(0), Some("x;y"));

        let tabs = write_file(&dir, "tabs.tsv", b"a\tb\n1\t2\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{tabs}', separator => E'\\t')"
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["a", "b"]);
        // The path may be an `E''` string too; forward slashes keep it free of escapes.
        let df = query(&format!(
            "SELECT * FROM read_csv(E'{}', separator => E'\\t')",
            tabs.replace('\\', "/")
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["a", "b"]);

        let quoted = write_file(&dir, "quoted.csv", b"a,b\n1,\"x\"\n");
        let df = query(&format!(
            "SELECT b FROM read_csv('{quoted}', quote_char => NULL)"
        ))
        .unwrap();
        assert_eq!(df.column("b").unwrap().str().unwrap().get(0), Some("\"x\""));
        // As in Python, an empty `quote_char` also turns quoting off.
        let df = query(&format!(
            "SELECT b FROM read_csv('{quoted}', quote_char => '')"
        ))
        .unwrap();
        assert_eq!(df.column("b").unwrap().str().unwrap().get(0), Some("\"x\""));

        let comments = write_file(&dir, "comments.csv", b"a,b\n# note\n1,2\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{comments}', comment_prefix => '#')"
        ))
        .unwrap();
        assert_eq!(df.height(), 1);

        let preamble = write_file(&dir, "preamble.csv", b"junk\njunk\na,b\n1,2\n3,4\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{preamble}', skip_rows => 2, skip_rows_after_header => 1)"
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["a", "b"]);
        assert_eq!(df.height(), 1);

        let unbalanced = write_file(&dir, "unbalanced.csv", b"title \"open\na,b\n1,2\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{unbalanced}', skip_lines => 1)"
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["a", "b"]);

        let invalid_utf8 = write_file(&dir, "lossy.csv", b"a\n\xff\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{invalid_utf8}', encoding => 'utf8-lossy')"
        ))
        .unwrap();
        assert_eq!(
            df.column("a").unwrap().str().unwrap().get(0),
            Some("\u{FFFD}")
        );
    }

    #[test]
    fn names_and_types() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "types.csv", b"a,b\n1,2\n");

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', new_columns => ['x', 'y'])"
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["x", "y"]);

        let headerless = write_file(&dir, "headerless.csv", b"1,2\n3,4\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{headerless}', has_header => false, schema => {{'x': 'INT', 'y': 'VARCHAR'}})"
        ))
        .unwrap();
        assert_eq!(df.get_column_names(), ["x", "y"]);
        assert_eq!(dtypes(&df), [DataType::Int32, DataType::String]);

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', schema_overrides => {{'b': 'VARCHAR'}})"
        ))
        .unwrap();
        assert_eq!(dtypes(&df), [DataType::Int64, DataType::String]);

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', schema_overrides => ['VARCHAR', 'DOUBLE'])"
        ))
        .unwrap();
        assert_eq!(dtypes(&df), [DataType::String, DataType::Float64]);

        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', infer_schema => false)"
        ))
        .unwrap();
        assert_eq!(dtypes(&df), [DataType::String, DataType::String]);

        let dates = write_file(&dir, "dates.csv", b"d\n2026-01-01\n");
        let df = query(&format!("SELECT * FROM read_csv('{dates}')")).unwrap();
        assert_eq!(dtypes(&df), [DataType::Date]);
        let df = query(&format!(
            "SELECT * FROM read_csv('{dates}', try_parse_dates => false)"
        ))
        .unwrap();
        assert_eq!(dtypes(&df), [DataType::String]);

        let decimals = write_file(&dir, "decimals.csv", b"a;b\n1,5;2\n");
        let df = query(&format!(
            "SELECT a FROM read_csv('{decimals}', separator => ';', decimal_comma => true)"
        ))
        .unwrap();
        assert_eq!(df.column("a").unwrap().f64().unwrap().get(0), Some(1.5));
    }

    #[test]
    fn null_values_and_errors() {
        let dir = tempfile::tempdir().unwrap();
        let single = write_file(&dir, "na.csv", b"a,b\nNA,1\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{single}', null_values => 'NA')"
        ))
        .unwrap();
        assert_eq!(df.column("a").unwrap().null_count(), 1);

        let several = write_file(&dir, "na_dash.csv", b"a,b\nNA,-\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{several}', null_values => ['NA', '-'])"
        ))
        .unwrap();
        assert_eq!(
            df.columns().iter().map(|c| c.null_count()).sum::<usize>(),
            2
        );

        let per_column = write_file(&dir, "na_per_column.csv", b"a,b\nNA,NA\n");
        let df = query(&format!(
            "SELECT * FROM read_csv('{per_column}', null_values => {{'a': 'NA'}})"
        ))
        .unwrap();
        assert_eq!(df.column("a").unwrap().null_count(), 1);
        assert_eq!(df.column("b").unwrap().str().unwrap().get(0), Some("NA"));

        let bad_int = write_file(&dir, "bad_int.csv", b"a\n1\nx\n");
        let sql =
            format!("SELECT * FROM read_csv('{bad_int}', schema_overrides => {{'a': 'INT'}})");
        assert!(query(&sql).is_err());
        let df = query(&format!(
            "SELECT * FROM read_csv('{bad_int}', schema_overrides => {{'a': 'INT'}}, ignore_errors => true)"
        ))
        .unwrap();
        assert_eq!(df.column("a").unwrap().null_count(), 1);
    }

    #[test]
    fn ragged_and_multiple_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "too_wide.csv", b"a,b\n1,2,3\n");
        assert!(query(&format!("SELECT * FROM read_csv('{path}')")).is_err());
        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', truncate_ragged_lines => true)"
        ))
        .unwrap();
        assert_eq!(df.shape(), (1, 2));

        write_file(&dir, "part_1.csv", b"a\n1\n");
        let part_2 = write_file(&dir, "part_2.csv", b"a\n2\n");
        let glob = part_2.replace("part_2.csv", "part_*.csv");
        let df = query(&format!(
            "SELECT * FROM read_csv('{glob}', include_file_paths => 'file')"
        ))
        .unwrap();
        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.column("file").unwrap().n_unique().unwrap(), 2);
    }

    #[test]
    fn empty_file_returns_schema_when_headerless() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "empty.csv", b"");
        assert!(query(&format!("SELECT * FROM read_csv('{path}')")).is_err());
        let df = query(&format!(
            "SELECT * FROM read_csv('{path}', has_header => false, schema => {{'x': 'INT'}})"
        ))
        .unwrap();
        assert_eq!(df.shape(), (0, 1));
    }

    #[test]
    fn option_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "errors.csv", b"a,b\n1,2\n");
        let cases = [
            (
                "header => false",
                "unknown `read_csv` option 'header'; valid options are: has_header, separator",
            ),
            (
                "n_rows => 5",
                "`read_csv` option 'n_rows' is not supported in SQL: use LIMIT",
            ),
            (
                "missing_columns => 'insert'",
                "not supported in SQL: it is unstable; for rows of different widths, use infer_schema_length => NULL",
            ),
            (
                "has_header = false",
                "`read_csv` options must be named with '=>' (found: has_header = false)",
            ),
            (
                "false",
                "`read_csv` options must be named with '=>' (found: false)",
            ),
            (
                "has_header => false, HAS_HEADER => true",
                "`read_csv` option 'has_header' is given more than once",
            ),
            (
                "has_header => 'yes'",
                "`read_csv` option 'has_header' expects a boolean; found: 'yes'",
            ),
            (
                "skip_rows => -1",
                "`read_csv` option 'skip_rows' expects a non-negative integer",
            ),
            (
                "separator => '\\t'",
                "expects a single-byte character; found: '\\t'; escape sequences need an E'' string",
            ),
            (
                "schema => {'x': 'INT('}",
                "`read_csv` option 'schema' has an invalid type name: 'INT('",
            ),
            (
                "schema => ['INT']",
                "`read_csv` option 'schema' expects a dictionary",
            ),
            (
                "has_header => *",
                "`read_csv` option 'has_header' expects a value; found: *",
            ),
            (
                "encoding => 'latin1'",
                "`read_csv` option 'encoding' expects 'utf8' or 'utf8-lossy'; found: 'latin1'",
            ),
        ];
        for (options, expected) in cases {
            let err = query_err(&format!("SELECT * FROM read_csv('{path}', {options})"));
            assert!(err.contains(expected), "{options}: {err}");
        }

        // Excluded options without an alternative get no trailing reason.
        let err = query_err(&format!("SELECT * FROM read_csv('{path}', glob => false)"));
        assert!(
            err.contains("`read_csv` option 'glob' is not supported in SQL")
                && !err.contains("SQL:"),
            "{err}"
        );
        let err = query_err("SELECT * FROM read_csv()");
        assert!(
            err.contains("`read_csv` expects a file path as its first argument"),
            "{err}"
        );
        let err = query_err("SELECT * FROM read_csv(has_header => false)");
        assert!(
            err.contains("`read_csv` expects a file path as its first argument"),
            "{err}"
        );
    }
}
