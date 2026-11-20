//! Tests of the command-line contract: what `polars` writes to stdout and stderr, and its exit
//! code. Text that clap words, such as the layout of the help, is not compared in full.

use std::io::Write;
use std::process::Stdio;

mod common;

const FORMATS: [&str; 5] = ["table", "csv", "ndjson", "parquet", "arrow-stream"];

/// The result of [`by_category`] as CSV.
const BY_CATEGORY: &str = "\
category,n,calories
fruit,7,410
meat,5,540
seafood,8,1250
vegetables,7,192
";

fn by_category(source: &str) -> String {
    format!(
        "SELECT category, COUNT(*) AS n, SUM(calories) AS calories FROM {source} \
         GROUP BY category ORDER BY category"
    )
}

/// A new directory for the files of test `name`. Its path has `/` as the separator, so that it
/// can be written in SQL on every platform.
fn temp_dir(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("polars_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_str().unwrap().replace('\\', "/")
}

/// Runs the command with nothing on standard input. Returns the exit code, stdout and stderr.
fn run(args: &[&str]) -> (i32, String, String) {
    run_with_stdin(args, "")
}

fn run_with_stdin(args: &[&str], stdin: &str) -> (i32, String, String) {
    let mut child = common::polars()
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Dropping the handle closes the input.
    let mut input = child.stdin.take().unwrap();
    input.write_all(stdin.as_bytes()).unwrap();
    drop(input);

    let output = child.wait_with_output().unwrap();
    (
        output.status.code().expect("the command was killed"),
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

#[test]
fn a_result_is_printed_as_a_table() {
    let sql = by_category("read_csv('../../examples/datasets/foods1.csv')");
    let (code, out, err) = run(&["-c", &sql]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.starts_with("shape: (4, 3)\n"), "{out}");
    assert!(out.contains("│ seafood    ┆ 8   ┆ 1250     │"), "{out}");
}

#[test]
fn every_file_format_can_be_read() {
    for source in [
        "read_csv('../../examples/datasets/foods1.csv')",
        "read_ipc('../../py-polars/tests/unit/io/files/foods1.ipc')",
        "read_ndjson('../../py-polars/tests/unit/io/files/foods1.ndjson')",
        "read_parquet('../../py-polars/tests/unit/io/files/foods1.parquet')",
    ] {
        let (code, out, err) = run(&["--format", "csv", "-c", &by_category(source)]);
        assert_eq!(
            (code, out.as_str(), err.as_str()),
            (0, BY_CATEGORY, ""),
            "{source}"
        );
    }
}

#[test]
fn ndjson_output() {
    let sql = "SELECT category, calories, sugars_g \
               FROM read_csv('../../examples/datasets/foods1.csv') \
               ORDER BY calories DESC, category, sugars_g LIMIT 3";
    let expected = "\
{\"category\":\"seafood\",\"calories\":200,\"sugars_g\":0}
{\"category\":\"seafood\",\"calories\":200,\"sugars_g\":2}
{\"category\":\"seafood\",\"calories\":200,\"sugars_g\":2}
";
    assert_eq!(
        run(&["--format", "ndjson", "-c", sql]),
        (0, expected.to_string(), String::new())
    );
}

#[test]
fn an_empty_result_keeps_its_columns() {
    let (code, out, err) = run(&["-c", "SELECT 1 AS a WHERE false"]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out.starts_with("shape: (0, 1)\n"), "{out}");
    assert!(out.contains("│ a   │"), "{out}");

    assert_eq!(
        run(&[
            "--format",
            "csv",
            "-c",
            "SELECT 1 AS a, 'x' AS b WHERE false"
        ]),
        (0, "a,b\n".to_string(), String::new())
    );
}

/// `INITCAP` needs the `nightly` feature and a join on `<` needs `iejoin`.
#[test]
fn sql_of_the_python_runtime_is_built_in() {
    assert_eq!(
        run(&[
            "--format",
            "csv",
            "-c",
            "SELECT INITCAP('hello world') AS s"
        ]),
        (0, "s\nHello World\n".to_string(), String::new())
    );

    let join = "WITH a AS (SELECT * FROM (VALUES (1), (5)) AS t (x)), \
                     b AS (SELECT * FROM (VALUES (2), (4), (6)) AS t (y)) \
                SELECT x, y FROM a JOIN b ON a.x < b.y ORDER BY x, y";
    assert_eq!(
        run(&["--format", "csv", "-c", join]),
        (0, "x,y\n1,2\n1,4\n1,6\n5,6\n".to_string(), String::new())
    );
}

#[test]
fn a_failing_statement_reports_on_stderr() {
    assert_eq!(
        run(&["-c", "SELECT LEN('x')"]),
        (
            1,
            String::new(),
            "SQLInterfaceError: unsupported function 'len'
 --> <command>:1:8
  |
1 | SELECT LEN('x')
  |        ^^^^^^^^
  = hint: use LENGTH instead
"
            .to_string()
        )
    );
}

#[test]
fn a_syntax_error_is_located_in_the_script() {
    let (code, _, err) = run(&["-c", "SELECT 1;\nSELECT 1 +* 2"]);
    assert_eq!(code, 1);
    assert!(err.contains("\n --> <command>:2:11\n"), "{err}");
}

#[test]
fn a_script_file_runs_in_one_session() {
    let expected = "\
category,calories
fruit,410
meat,540
seafood,1250
vegetables,192
n
7
";
    assert_eq!(
        run(&["--format", "csv", "-f", "tests/data/session.sql"]),
        (0, expected.to_string(), String::new())
    );

    let (code, out, err) = run(&["-f", "tests/data/session.sql"]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert_eq!(out.matches("shape: ").count(), 2, "{out}");
}

#[test]
fn effect_statements_print_nothing() {
    // They still take effect.
    let script = "CREATE TABLE t AS SELECT 1 AS a UNION ALL SELECT 2; \
                  CREATE VIEW v AS SELECT 3 AS b; DELETE FROM t WHERE a = 2; \
                  SELECT COUNT(*) AS n FROM t; SELECT b FROM v; \
                  TRUNCATE TABLE t; SELECT COUNT(*) AS n FROM t; DROP TABLE t";
    assert_eq!(
        run(&["--format", "csv", "-c", script]),
        (0, "n\n1\nb\n3\nn\n0\n".to_string(), String::new())
    );
    let (code, out, err) = run(&["-c", script]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert_eq!(out.matches("shape: ").count(), 3, "{out}");
    // What the engine answers them with is not computed: here, the rows that `DELETE` leaves.
    let script = "CREATE TABLE t AS SELECT 'x' AS a; DELETE FROM t WHERE CAST(a AS INT) = 1; \
                  SELECT 42 AS n";
    assert_eq!(
        run(&["--format", "csv", "-c", script]),
        (0, "n\n42\n".to_string(), String::new())
    );

    // A format that writes one file takes effect statements before the one result.
    let script = "CREATE TABLE t AS SELECT 1 AS a; SELECT * FROM t";
    let output = common::polars()
        .args(["--format", "parquet", "-c", script])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.starts_with(b"PAR1") && output.stderr.is_empty());
    // Without a result, nothing runs, so `COPY` writes no file.
    let dir = temp_dir("effect");
    for format in ["parquet", "arrow-stream"] {
        let copy = format!("COPY (SELECT 1 AS a) TO '{dir}/a.csv'");
        let (code, out, _) = run(&["--format", format, "-c", &copy]);
        assert_eq!((code, out.as_str()), (2, ""), "{format}");
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir_all(dir).unwrap();
    // `COPY ... TO STDOUT` has rows, in its own format.
    let copy = "COPY (SELECT 1 AS a) TO STDOUT (FORMAT csv)";
    assert_eq!(
        run(&["--format", "parquet", "-c", copy]),
        (0, "a\n1\n".to_string(), String::new())
    );
}

#[test]
fn copy_writes_a_file() {
    let dir = temp_dir("copy");
    // The file is there for the next statement.
    let script = format!(
        "COPY (SELECT category, calories FROM read_csv('../../examples/datasets/foods1.csv')) \
         TO '{dir}/foods.parquet'; {}",
        by_category(&format!("'{dir}/foods.parquet'"))
    );
    assert_eq!(
        run(&["--format", "csv", "-c", &script]),
        (0, BY_CATEGORY.to_string(), String::new())
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn read_only_refuses_to_write_a_file() {
    let dir = temp_dir("read_only");
    let copy = format!("COPY (SELECT 1 AS a) TO '{dir}/a.csv'");

    let (code, out, err) = run(&["--read-only", "-c", &copy]);
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.contains("writing files is disabled by --read-only"),
        "{err}"
    );

    // The environment variable does the same.
    let output = common::polars()
        .args(["-c", &copy])
        .env("POLARS_READ_ONLY", "1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(
        err.contains("writing files is disabled by POLARS_READ_ONLY"),
        "{err}"
    );
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);

    // The session and reading files are not limited.
    let script = "CREATE TABLE foods AS SELECT * FROM '../../examples/datasets/foods1.csv'; \
                  SELECT COUNT(*) AS n FROM foods";
    assert_eq!(
        run(&["--read-only", "--format", "csv", "-c", script]),
        (0, "n\n27\n".to_string(), String::new())
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn copy_to_stdout_writes_rows_in_its_own_format() {
    let rows = by_category("read_csv('../../examples/datasets/foods1.csv')");

    // Only the rows, whatever `--format` is. No file is written, so `--read-only` allows it.
    assert_eq!(
        run(&[
            "--read-only",
            "-c",
            &format!("COPY ({rows}) TO STDOUT (FORMAT csv)")
        ]),
        (0, BY_CATEGORY.to_string(), String::new())
    );

    // A table of the session, with the options of the format, between other results.
    let script = format!(
        "CREATE TABLE foods AS {rows}; \
         COPY foods (calories, category) TO STDOUT (FORMAT csv, HEADER false, DELIMITER '|'); \
         SELECT COUNT(*) AS n FROM foods"
    );
    let expected = "\
410|fruit
540|meat
1250|seafood
192|vegetables
n
4
";
    assert_eq!(
        run(&["--format", "csv", "-c", &script]),
        (0, expected.to_string(), String::new())
    );

    // A binary format is one file, which does not share the output with another result.
    let script = "COPY (SELECT 1 AS a) TO STDOUT (FORMAT parquet); SELECT 2 AS b";
    let (code, out, _) = run(&["-c", script]);
    assert_eq!((code, out.as_str()), (1, ""));
    let script = "CREATE TABLE t AS SELECT 1 AS a; COPY t TO STDOUT (FORMAT parquet)";
    let output = common::polars().args(["-c", script]).output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.starts_with(b"PAR1"));
}

#[test]
fn a_script_is_read_from_standard_input() {
    let script = "SELECT 1 AS a;\nSELECT 'x;y' AS b;\n";
    assert_eq!(
        run_with_stdin(&["--format", "csv"], script),
        (0, "a\n1\nb\nx;y\n".to_string(), String::new())
    );
}

#[test]
fn a_script_stops_at_the_first_failure() {
    let script = "SELECT 1 AS a; SELECT * FROM missing; SELECT 2 AS b";
    let (code, out, err) = run(&["--format", "csv", "-c", script]);
    assert_eq!((code, out.as_str()), (1, "a\n1\n"));
    assert!(
        err.starts_with("statement 2: SQLInterfaceError: relation 'missing' was not found\n"),
        "{err}"
    );
}

#[test]
fn an_unclosed_string_fails_after_the_statements_before_it() {
    let (code, out, err) = run(&["--format", "csv", "-c", "SELECT 1 AS a; SELECT 'abc"]);
    assert_eq!((code, out.as_str()), (1, "a\n1\n"));
    assert!(
        err.starts_with("statement 2: SQLInterfaceError: sql parser error: Unterminated string"),
        "{err}"
    );
    // A format that writes one file runs none of them.
    let (code, out, err) = run(&["--format", "parquet", "-c", "SELECT 1 AS a; SELECT 'abc"]);
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(err.starts_with("statement 2: "), "{err}");
}

#[test]
fn usage_errors() {
    // The start of the message; an empty one stands for clap's own wording.
    let cases: [(&[&str], &str); 6] = [
        (&[], "no SQL was given\n"),
        (
            &["--format", "parquet", "-c", "SELECT 1 AS a; SELECT 2 AS b"],
            "format 'parquet' writes one file, so the last statement must be the only one that \
             returns rows\n",
        ),
        (
            &[
                "--format",
                "arrow-stream",
                "-c",
                "CREATE TABLE t AS SELECT 1 AS a",
            ],
            "format 'arrow-stream' writes one file, so the last statement must be the only one \
             that returns rows\n",
        ),
        (&["-c", "SELECT 1", "-f", "tests/data/session.sql"], ""),
        (&["--format", "xml", "-c", "SELECT 1"], ""),
        (&["--unknown"], ""),
    ];
    for (args, message) in cases {
        let (code, out, err) = run(args);
        assert_eq!((code, out.as_str()), (2, ""), "{args:?}");
        assert!(
            err.starts_with(&format!("error: {message}")),
            "{args:?}: {err}"
        );
    }
}

#[test]
fn an_unknown_format_is_answered_with_the_formats() {
    let (_, _, err) = run(&["--format", "xml", "-c", "SELECT 1"]);
    for format in FORMATS {
        assert!(err.contains(format), "{format}: {err}");
    }
}

#[test]
fn help() {
    let (code, out, err) = run(&["--help"]);
    assert_eq!((code, err.as_str()), (0, ""));
    let parts = [
        "Usage: polars [OPTIONS]",
        "-c, --command <SQL>",
        "-f, --file <FILE>",
        "--format <FMT>",
        "--read-only",
        "Exit codes:",
        "This command is unstable",
    ];
    for part in parts.into_iter().chain(FORMATS) {
        assert!(out.contains(part), "{part}: {out}");
    }
}

/// The reason is worded by the operating system.
#[test]
fn unreadable_script() {
    let (code, out, err) = run(&["-f", "tests/data/no-such-file.sql"]);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(
        err.starts_with("error: cannot read 'tests/data/no-such-file.sql': "),
        "{err}"
    );
}

#[test]
fn version() {
    assert_eq!(
        run(&["--version"]),
        (
            0,
            format!("polars {}\n", env!("CARGO_PKG_VERSION")),
            String::new()
        )
    );
}
