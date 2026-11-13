//! Command-line SQL engine for Polars.
//!
//! This crate is unstable: the command line, the output and this library can change in any
//! release.
//!
//! The binary only parses arguments, reads the script and sets up the printer. Everything that
//! runs SQL goes through [`run`], which reports what happens as [`Event`]s.

use std::any::Any;
use std::io::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

use polars::error::abort::{QueryAborted, catch_polars_abort};
use polars::prelude::{DataFrame, Engine, PolarsResult};
use polars::sql::SQLContext;

mod args;
mod error;
mod printer;
mod script;

pub use args::{Args, Input};
pub use error::{ErrorRecord, Location};
pub use printer::{Event, Printer};
pub use script::Script;

/// The exit codes of the `polars` command.
pub mod exit_code {
    pub const OK: u8 = 0;
    /// A statement failed, or its result could not be written.
    pub const FAILURE: u8 = 1;
    /// The command line could not be used.
    pub const USAGE: u8 = 2;
    /// A statement panicked, which is a bug.
    pub const INTERNAL_ERROR: u8 = 101;
    pub const INTERRUPTED: u8 = 130;
}

/// Runs the statements of `script` in order, in one session, and reports the outcome to
/// `printer`. Stops at the first statement that fails. Returns the exit code.
pub fn run<O: Write, E: Write>(script: &Script, printer: &mut Printer<O, E>) -> u8 {
    let mut context = SQLContext::new();
    let code = run_statements(script, printer, |sql| {
        let plan = context.execute(sql)?;
        Ok(plan.collect_with_engine(Engine::Streaming)?.unwrap_single())
    });
    drop(context);
    // Spill files are deleted in the background; wait for that before the process can exit.
    polars_ooc::flush_ooc_cleanup();
    code
}

fn run_statements<O: Write, E: Write>(
    script: &Script,
    printer: &mut Printer<O, E>,
    mut execute: impl FnMut(&str) -> PolarsResult<DataFrame>,
) -> u8 {
    // An interrupt is only recorded while something catches it. Catching around the whole run
    // also records one that arrives between statements; the next statement then stops at once.
    let outcome = catch_polars_abort(AssertUnwindSafe(|| {
        run_until_failure(script, printer, &mut execute)
    }));
    match outcome {
        Ok(Ok(code)) => code,
        // The output cannot be written to.
        Ok(Err(_)) => exit_code::FAILURE,
        Err(QueryAborted::KeyboardInterrupt) => {
            let _ = printer.event(Event::Interrupted);
            exit_code::INTERRUPTED
        },
        Err(QueryAborted::OocOutOfDisk) => exit_code::FAILURE,
    }
}

/// Returns the exit code; `Err` means the printer failed.
fn run_until_failure<O: Write, E: Write>(
    script: &Script,
    printer: &mut Printer<O, E>,
    execute: &mut impl FnMut(&str) -> PolarsResult<DataFrame>,
) -> io::Result<u8> {
    printer.event(Event::Start {
        statements: script.len(),
    })?;

    for (index, sql) in script.statements().enumerate() {
        let number = index + 1;
        // Printing is inside both guards: rendering a result can panic, and writing it can
        // abort the query as running it can.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            catch_polars_abort(AssertUnwindSafe(|| -> io::Result<PolarsResult<()>> {
                let result = match execute(sql) {
                    Ok(result) => result,
                    Err(error) => return Ok(Err(error)),
                };
                printer.event(Event::Schema(result.schema().clone()))?;
                printer.event(Event::Rows(result))?;
                printer.event(Event::StatementEnd)?;
                Ok(Ok(()))
            }))
        }));
        let (event, code) = match outcome {
            Ok(Ok(Ok(Ok(())))) => continue,
            Ok(Ok(Err(output_error))) => return Err(output_error),
            Ok(Ok(Ok(Err(error)))) => (
                Event::Error(ErrorRecord::from_polars(&error, number)),
                exit_code::FAILURE,
            ),
            Ok(Err(QueryAborted::KeyboardInterrupt)) => {
                (Event::Interrupted, exit_code::INTERRUPTED)
            },
            Ok(Err(QueryAborted::OocOutOfDisk)) => (
                Event::Error(ErrorRecord::out_of_disk(number)),
                exit_code::FAILURE,
            ),
            Err(panic) => (
                Event::Error(ErrorRecord::internal(panic_message(&*panic), number)),
                exit_code::INTERNAL_ERROR,
            ),
        };
        // The exit code stands also when the report cannot be written, or when writing it
        // panics as printing the result did.
        let _ = catch_unwind(AssertUnwindSafe(|| printer.event(event)));
        return Ok(code);
    }

    if let Some(error) = script.split_error() {
        printer.event(Event::Error(error))?;
        return Ok(exit_code::FAILURE);
    }
    Ok(exit_code::OK)
}

fn panic_message(panic: &(dyn Any + Send)) -> &str {
    if let Some(message) = panic.downcast_ref::<&str>() {
        message
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message
    } else {
        "unknown panic"
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{DataType, IDX_DTYPE, df, polars_bail};

    use super::*;

    fn captured(
        run: impl FnOnce(&mut Printer<&mut Vec<u8>, &mut Vec<u8>>) -> u8,
    ) -> (u8, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(&mut Printer::new(&mut out, &mut err));
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    fn run_captured(sql: &str) -> (u8, String, String) {
        let script = Script::new(sql.to_string());
        captured(|printer| run(&script, printer))
    }

    /// Runs `sql` with `execute` in place of the engine.
    fn run_with(
        sql: &str,
        execute: impl FnMut(&str) -> PolarsResult<DataFrame>,
    ) -> (u8, String, String) {
        let script = Script::new(sql.to_string());
        captured(|printer| run_statements(&script, printer, execute))
    }

    #[test]
    fn row_index_is_64_bit() {
        assert_eq!(IDX_DTYPE, DataType::UInt64);
    }

    #[test]
    fn result_goes_to_stdout() {
        let (code, out, err) = run_captured("SELECT 1 AS a");
        assert_eq!(code, exit_code::OK);
        assert!(out.starts_with("shape: (1, 1)\n"), "{out}");
        assert_eq!(err, "");
    }

    #[test]
    fn error_goes_to_stderr() {
        let (code, out, err) = run_captured("SELECT * FROM missing");
        assert_eq!(code, exit_code::FAILURE);
        assert_eq!(out, "");
        assert!(err.starts_with("SQLInterfaceError: "), "{err}");
    }

    #[test]
    fn statements_share_a_session() {
        let (code, out, err) =
            run_captured("CREATE TABLE t AS SELECT 7 AS a; SELECT a + 1 AS b FROM t");
        assert_eq!((code, err.as_str()), (exit_code::OK, ""));
        assert!(out.contains("│ 8   │"), "{out}");
    }

    #[test]
    fn an_empty_script_succeeds() {
        assert_eq!(
            run_captured(" ; -- nothing\n"),
            (exit_code::OK, String::new(), String::new())
        );
    }

    #[test]
    fn the_run_stops_at_the_first_failure() {
        let mut executed = Vec::new();
        let (code, out, err) = run_with("first; second; third", |sql| {
            executed.push(sql.to_string());
            if sql == "second" {
                polars_bail!(ComputeError: "no good");
            }
            df!("a" => [1])
        });

        assert_eq!(executed, ["first", "second"]);
        assert_eq!(code, exit_code::FAILURE);
        assert!(out.starts_with("shape: (1, 1)\n"), "{out}");
        assert_eq!(err, "statement 2: ComputeError: no good\n");
    }

    #[test]
    fn a_statement_that_cannot_be_split_fails_last() {
        let mut executed = Vec::new();
        let (code, _, err) = run_with("first; SELECT 'abc", |sql| {
            executed.push(sql.to_string());
            df!("a" => [1])
        });

        assert_eq!(executed, ["first"]);
        assert_eq!(code, exit_code::FAILURE);
        assert!(
            err.starts_with("statement 2: SQLInterfaceError: sql parser error: Unterminated"),
            "{err}"
        );
    }

    #[test]
    fn a_panic_is_an_internal_error() {
        let mut executed = 0;
        let (code, out, err) = run_with("first; second", |_| {
            executed += 1;
            panic!("boom");
        });

        assert_eq!(executed, 1);
        assert_eq!(code, exit_code::INTERNAL_ERROR);
        assert_eq!(out, "");
        assert_eq!(
            err,
            "statement 1: InternalError: boom (this is a bug in Polars, please report it)\n"
        );
    }

    #[test]
    fn rows_before_an_error_are_not_printed() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut printer = Printer::new(&mut out, &mut err);
        let rows = df!("a" => [1]).unwrap();

        printer.event(Event::Schema(rows.schema().clone())).unwrap();
        printer.event(Event::Rows(rows)).unwrap();
        let error = ErrorRecord {
            class: "ComputeError",
            message: "failed halfway".to_string(),
            hint: None,
            location: None,
            statement: 1,
        };
        printer.event(Event::Error(error)).unwrap();

        assert_eq!(out, b"");
        assert_eq!(err, b"ComputeError: failed halfway\n");
    }
}
