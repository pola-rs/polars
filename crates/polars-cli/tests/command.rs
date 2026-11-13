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
        run(&["-c", "SELECT * FROM missing"]),
        (
            1,
            String::new(),
            "SQLInterfaceError: relation 'missing' was not found\n".to_string()
        )
    );

    let (code, out, err) = run(&["-c", "SELECT 1 +"]);
    assert_eq!((code, out.as_str()), (1, ""));
    assert!(
        err.starts_with("SQLInterfaceError: sql parser error: "),
        "{err}"
    );
}

#[test]
fn a_script_file_runs_in_one_session() {
    let expected = "\
Response
CREATE TABLE foods
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
    assert_eq!(out.matches("shape: ").count(), 3, "{out}");
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
    assert_eq!(
        run(&["--format", "csv", "-c", script]),
        (
            1,
            "a\n1\n".to_string(),
            "statement 2: SQLInterfaceError: relation 'missing' was not found\n".to_string()
        )
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
}

#[test]
fn usage_errors() {
    // The start of the message; an empty one stands for clap's own wording.
    let cases: [(&[&str], &str); 5] = [
        (&[], "no SQL was given\n"),
        (
            &["--format", "parquet", "-c", "SELECT 1 AS a; SELECT 2 AS b"],
            "format 'parquet' writes one file, so it takes a script of one statement\n",
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
