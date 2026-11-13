//! Golden tests of the command-line contract: stdout, stderr and the exit code.
//!
//! A case is a pair of files in `tests/golden`. `<name>.args` holds the arguments, one per line.
//! `<name>.expected` holds what the command must produce. The command runs in the crate
//! directory, so paths in the arguments are relative to it.
//!
//! Run with `POLARS_CLI_BLESS=1` to write the `.expected` files from the actual output.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn run_case(args_path: &Path) -> String {
    let args = fs::read_to_string(args_path).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_polars"));
    command
        .args(args.lines())
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    // Settings of the environment must not leak into the expected output.
    for (name, _) in env::vars_os() {
        // Names of environment variables are case-insensitive on Windows.
        if name
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("POLARS_")
        {
            command.env_remove(name);
        }
    }
    let output = command.output().unwrap();

    let mut actual = String::new();
    match output.status.code() {
        Some(code) => writeln!(actual, "exit: {code}").unwrap(),
        None => writeln!(actual, "exit: none ({})", output.status).unwrap(),
    }
    for (name, stream) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        writeln!(actual, "--- {name} ---").unwrap();
        actual.push_str(&String::from_utf8_lossy(stream));
    }
    normalize(&actual)
}

/// Git may check the expected files out with other line endings than the command writes.
fn normalize(text: &str) -> String {
    text.replace("\r\n", "\n")
}

#[test]
fn golden() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bless = env::var_os("POLARS_CLI_BLESS").is_some();

    let mut cases: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "args"))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no cases in {}", dir.display());

    let mut failures = Vec::new();
    for args_path in cases {
        let name = args_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let expected_path = args_path.with_extension("expected");
        let actual = run_case(&args_path);

        if bless {
            fs::write(&expected_path, &actual).unwrap();
            continue;
        }
        let expected = fs::read_to_string(&expected_path)
            .map(|text| normalize(&text))
            .unwrap_or_default();
        if actual != expected {
            failures.push(format!(
                "case '{name}'\n=== expected ===\n{expected}\n=== actual ===\n{actual}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} golden case(s) differ; run with POLARS_CLI_BLESS=1 to accept the actual output\n\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The version changes with every release, so it is not a golden case.
#[test]
fn version() {
    let output = Command::new(env!("CARGO_BIN_EXE_polars"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("polars {}\n", env!("CARGO_PKG_VERSION"))
    );
}
