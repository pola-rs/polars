//! Runs the `.slt` files in `slt/` on polars-sql and checks the failures against
//! `expected_failures.txt`.
//!
//! - `cargo run -p polars-sqllogictest [FILTER]...` runs the files whose path contains a filter.
//! - `cargo run -p polars-sqllogictest -- --queries FILE` runs the statements in FILE and prints
//!   the results as JSON lines (used by `tools/gen_window_slt.py`).
mod engine;
mod expected;
mod output;
mod query_mode;
mod setup;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sqllogictest::{DefaultColumnType, Record, RecordOutput, Runner, parse_file};

use crate::engine::{EngineError, PolarsEngine};
use crate::expected::ExpectedFailures;

/// Message of the last panic. Engine panics can happen on worker threads, so this is global.
static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

fn slt_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("slt")
}

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("expected_failures.txt")
}

fn collect_slt_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_slt_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "slt") {
            out.push(path);
        }
    }
}

fn record_line(record: &Record<DefaultColumnType>) -> Option<u32> {
    match record {
        Record::Query { loc, .. } | Record::Statement { loc, .. } => Some(loc.line()),
        _ => None,
    }
}

struct Report {
    passed: usize,
    passing_keys: Vec<String>,
    failures: Vec<(String, String)>,
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            info.to_string()
        };
        let loc = info
            .location()
            .map(|l| format!(" at {}:{}", l.file(), l.line()))
            .unwrap_or_default();
        LAST_PANIC
            .lock()
            .unwrap()
            .get_or_insert(format!("{msg}{loc}"));
    }));
}

/// Runs `f`, turning a panic into an error message.
fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    *LAST_PANIC.lock().unwrap() = None;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|_| {
        let msg = LAST_PANIC.lock().unwrap().take().unwrap_or_default();
        format!("PANIC: {msg}")
    })
}

fn run_file(path: &Path, rel: &str, report: &mut Report) {
    let records = match parse_file::<DefaultColumnType>(path) {
        Ok(records) => records,
        Err(err) => {
            report
                .failures
                .push((rel.to_string(), format!("failed to parse file: {err}")));
            return;
        },
    };

    let mut runner = Runner::new(|| std::future::ready(Ok::<_, EngineError>(PolarsEngine::new())));
    runner.add_label("polars");

    for record in records {
        let Some(line) = record_line(&record) else {
            let _ = runner.run(record);
            continue;
        };
        let key = format!("{rel}:{line}");
        match catch_panic(|| runner.run(record)) {
            Ok(Ok(RecordOutput::Nothing)) => {},
            Ok(Ok(_)) => {
                report.passed += 1;
                report.passing_keys.push(key);
            },
            Ok(Err(err)) => report.failures.push((key, format!("{err}"))),
            Err(msg) => report.failures.push((key, msg)),
        }
    }
}

fn main() {
    install_panic_hook();

    let mut args = std::env::args().skip(1);
    let mut filters = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--queries" {
            let path = args.next().expect("--queries needs a file path");
            query_mode::run(Path::new(&path));
            return;
        }
        if !arg.starts_with('-') {
            filters.push(arg);
        }
    }

    let root = slt_root();
    let mut files = Vec::new();
    collect_slt_files(&root, &mut files);
    files.sort();

    let mut report = Report {
        passed: 0,
        passing_keys: Vec::new(),
        failures: Vec::new(),
    };

    for path in &files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if !filters.is_empty() && !filters.iter().any(|f| rel.contains(f.as_str())) {
            continue;
        }
        run_file(path, &rel, &mut report);
    }

    let baseline = match ExpectedFailures::load(&baseline_path()) {
        Ok(baseline) => baseline,
        Err(err) => {
            eprintln!("failed to load expected_failures.txt: {err}");
            std::process::exit(2);
        },
    };

    let mut unexpected_failures = Vec::new();
    let mut expected_fail = 0usize;
    for (key, message) in &report.failures {
        if baseline.contains(key) {
            expected_fail += 1;
        } else {
            unexpected_failures.push((key.clone(), message.clone()));
        }
    }

    let passing: std::collections::BTreeSet<&str> =
        report.passing_keys.iter().map(String::as_str).collect();
    let newly_passing: Vec<&String> = baseline
        .iter()
        .filter(|entry| passing.contains(entry.as_str()))
        .collect();

    for (key, message) in &unexpected_failures {
        eprintln!("FAIL {key}");
        eprintln!("{message}");
        eprintln!();
    }

    for key in &newly_passing {
        eprintln!(
            "PASS (unexpected) {key}\n  this entry now passes; remove it from expected_failures.txt\n"
        );
    }

    let total = report.passed + report.failures.len();
    let pass_rate = if total == 0 {
        100.0
    } else {
        report.passed as f64 / total as f64 * 100.0
    };
    println!(
        "polars-sqllogictest: {} passed / {} expected-fail / {} total ({:.1}% pass rate)",
        report.passed, expected_fail, total, pass_rate
    );

    if !unexpected_failures.is_empty() || !newly_passing.is_empty() {
        std::process::exit(1);
    }
}
