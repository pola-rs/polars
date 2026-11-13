//! Tests of how the command behaves as a process, which the golden cases cannot express.

use std::process::{Child, ExitStatus, Stdio};
use std::time::{Duration, Instant};

mod common;

/// Generous, so that a slow machine does not fail a test; a hang still does.
const LIMIT: Duration = Duration::from_secs(120);

/// Ends the command when a test fails before the command has finished.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Running {
    fn wait(&mut self) -> ExitStatus {
        let start = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(
                start.elapsed() < LIMIT,
                "the command did not finish within {LIMIT:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// Polars writes its verbose output from worker threads while the run waits for them.
#[test]
fn verbose_output_does_not_block_the_run() {
    let child = common::polars()
        .args([
            "-c",
            "SELECT category, COUNT(*) AS n FROM read_csv('../../examples/datasets/foods1.csv') \
             GROUP BY category",
        ])
        .env("POLARS_VERBOSE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    assert_eq!(Running(child).wait().code(), Some(0));
}

/// Parquet and Arrow output is binary, so it is read back instead of compared as text.
#[test]
fn binary_output_reads_back() {
    use std::io::Cursor;

    use polars::prelude::*;

    let sql = "SELECT n, n * 2 AS twice, 'row ' || n AS label \
               FROM generate_series(1, 50000) AS t (n) ORDER BY n";
    let expected = df!(
        "n" => (1..=50_000i64).collect::<Vec<_>>(),
        "twice" => (1..=50_000i64).map(|n| n * 2).collect::<Vec<_>>(),
        "label" => (1..=50_000i64).map(|n| format!("row {n}")).collect::<Vec<_>>(),
    )
    .unwrap();

    for format in ["parquet", "arrow-stream"] {
        let output = common::polars()
            .args(["--format", format, "-c", sql])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(0), "{format}");
        assert!(output.stderr.is_empty(), "{format}");

        let bytes = Cursor::new(output.stdout);
        let read = match format {
            "parquet" => ParquetReader::new(bytes).finish(),
            _ => IpcStreamReader::new(bytes).finish(),
        }
        .unwrap();
        assert!(read.equals(&expected), "{format}: {read}");
    }
}

#[cfg(unix)]
#[test]
fn an_interrupt_stops_the_run() {
    use std::io::{BufRead, BufReader, Read};
    use std::process::Command;
    use std::sync::mpsc;

    // The second statement runs for minutes when nothing stops it.
    let script = "SELECT 1 AS a; \
        WITH a AS (SELECT * FROM generate_series(1, 60000) AS t (x)), \
             b AS (SELECT * FROM generate_series(1, 60000) AS t (y)) \
        SELECT SUM(x * y) AS s FROM a CROSS JOIN b; \
        SELECT 2 AS b";
    let mut child = common::polars()
        .args(["-c", script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let mut running = Running(child);

    // Reports once the first result has been printed, then collects what follows it.
    let (first_result_printed, first_result) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut stdout = BufReader::new(stdout);
        let mut line = String::new();
        while !line.contains('└') && stdout.read_line(&mut line).unwrap() != 0 {}
        first_result_printed.send(()).unwrap();
        let mut rest = String::new();
        stdout.read_to_string(&mut rest).unwrap();
        rest
    });
    first_result
        .recv_timeout(LIMIT)
        .expect("the first statement did not print its result");

    let interrupt = Command::new("kill")
        .args(["-INT", &running.0.id().to_string()])
        .status()
        .unwrap();
    assert!(interrupt.success());

    assert_eq!(running.wait().code(), Some(130));
    assert_eq!(
        reader.join().unwrap(),
        "",
        "no result may follow the interrupt"
    );
    let mut errors = String::new();
    stderr.read_to_string(&mut errors).unwrap();
    assert_eq!(errors, "interrupted\n");
}
