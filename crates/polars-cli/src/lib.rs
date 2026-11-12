//! Command-line SQL engine for Polars.
//!
//! This crate is unstable: the command line, the output and this library can change in any
//! release.
//!
//! The binary only parses arguments, reads the script and sets up the printer. Everything that
//! runs SQL goes through [`run`], which reports what happens as [`Event`]s.

use std::any::Any;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::mpsc::sync_channel;

use polars::error::abort::{QueryAborted, catch_polars_abort};
use polars::prelude::{DataFrame, Engine, LazyFrame, PlanCallback, PolarsError, PolarsResult};
use polars::sql::SQLContext;

mod args;
mod error;
mod printer;
mod script;

pub use args::{Args, Input};
pub use error::{ErrorRecord, Location};
pub use printer::{Event, Format, Printer};
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
    let code = run_statements(script, printer, |sql| context.execute(sql));
    drop(context);
    // Spill files are deleted in the background; wait for that before the process can exit.
    polars_ooc::flush_ooc_cleanup();
    code
}

/// Why a statement did not complete.
enum Stop {
    Error(PolarsError),
    Aborted(QueryAborted),
}

impl From<PolarsError> for Stop {
    fn from(error: PolarsError) -> Self {
        Self::Error(error)
    }
}

/// `plan` turns a statement into the plan of its result.
fn run_statements<O: Write, E: Write>(
    script: &Script,
    printer: &mut Printer<O, E>,
    mut plan: impl FnMut(&str) -> PolarsResult<LazyFrame>,
) -> u8 {
    // An interrupt is only recorded while something catches it. Catching around the whole run
    // also records one that arrives between statements; the next statement then stops at once.
    let outcome = catch_polars_abort(AssertUnwindSafe(|| {
        run_until_failure(script, printer, &mut plan)
    }));
    match outcome {
        Ok(code) => code,
        Err(QueryAborted::KeyboardInterrupt) => {
            let _ = printer.event(Event::Interrupted);
            exit_code::INTERRUPTED
        },
        Err(QueryAborted::OocOutOfDisk) => exit_code::FAILURE,
    }
}

fn run_until_failure<O: Write, E: Write>(
    script: &Script,
    printer: &mut Printer<O, E>,
    plan: &mut impl FnMut(&str) -> PolarsResult<LazyFrame>,
) -> u8 {
    if printer
        .event(Event::Start {
            statements: script.len(),
        })
        .is_err()
    {
        return exit_code::FAILURE;
    }

    for (index, sql) in script.statements().enumerate() {
        let number = index + 1;
        // Printing is inside both guards: rendering a result can panic, and writing it can
        // abort the query as running it can.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            catch_polars_abort(AssertUnwindSafe(|| -> Result<(), Stop> {
                stream(plan(sql)?, printer)
            }))
            .unwrap_or_else(|aborted| Err(Stop::Aborted(aborted)))
        }));
        let (event, code) = match outcome {
            Ok(Ok(())) => continue,
            // A reader that closed the output, as `head` does, is not worth a message.
            Ok(Err(Stop::Error(_))) if printer.output_is_closed() => return exit_code::FAILURE,
            Ok(Err(Stop::Error(error))) => (
                Event::Error(ErrorRecord::from_polars(&error, number)),
                exit_code::FAILURE,
            ),
            Ok(Err(Stop::Aborted(QueryAborted::KeyboardInterrupt))) => {
                (Event::Interrupted, exit_code::INTERRUPTED)
            },
            Ok(Err(Stop::Aborted(QueryAborted::OocOutOfDisk))) => (
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
        return code;
    }

    if let Some(error) = script.split_error() {
        let _ = printer.event(Event::Error(error));
        return exit_code::FAILURE;
    }
    exit_code::OK
}

/// Runs `plan` on the streaming engine and hands its result to `printer` batch by batch, in
/// order, so that the whole result is never held here.
fn stream<O: Write, E: Write>(plan: LazyFrame, printer: &mut Printer<O, E>) -> Result<(), Stop> {
    let mut unsunk = plan.clone();
    // One batch in flight: the engine waits while a batch is being written.
    let (send, receive) = sync_channel::<DataFrame>(1);
    let plan = plan.sink_batches(
        // Returning `true` stops the query: nobody is left to take its batches.
        PlanCallback::new(move |batch| Ok(send.send(batch).is_err())),
        true,
        None,
    )?;

    std::thread::scope(|scope| {
        let engine = std::thread::Builder::new()
            .name("polars-cli-query".to_string())
            .spawn_scoped(scope, move || {
                catch_polars_abort(AssertUnwindSafe(|| {
                    plan.collect_with_engine(Engine::Streaming)
                }))
            })
            .map_err(PolarsError::from)?;

        let mut has_rows = false;
        let mut printed = Ok(());
        for batch in &receive {
            if !has_rows {
                has_rows = true;
                printed = printer.event(Event::Schema(batch.schema().clone()));
            }
            if printed.is_ok() {
                printed = printer.event(Event::Rows(batch));
            }
            if printed.is_err() {
                break;
            }
        }
        // Without a receiver the engine stops at its next batch.
        drop(receive);

        match engine.join() {
            Ok(Ok(ran)) => {
                printed?;
                ran.map_err(without_sink_context)?;
            },
            Ok(Err(aborted)) => return Err(Stop::Aborted(aborted)),
            Err(panic) => resume_unwind(panic),
        }

        if !has_rows {
            // No batch arrived to take the columns from.
            printer.event(Event::Schema(unsunk.collect_schema()?))?;
        }
        printer.event(Event::StatementEnd)?;
        Ok(())
    })
}

/// Errors name the plan nodes they passed through. The outermost one is the batch sink that
/// `stream` adds, which is not part of the statement.
fn without_sink_context(error: PolarsError) -> PolarsError {
    match error {
        PolarsError::Context { error, msg } if &*msg == "'sink'" => *error,
        error => error,
    }
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
    use polars::prelude::{DataType, IDX_DTYPE, IntoLazy, df, polars_bail};

    use super::*;

    fn captured(
        format: Format,
        run: impl FnOnce(&mut Printer<&mut Vec<u8>, &mut Vec<u8>>) -> u8,
    ) -> (u8, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(&mut Printer::new(&mut out, &mut err, format));
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    fn run_captured(sql: &str) -> (u8, String, String) {
        run_formatted(sql, Format::Table)
    }

    fn run_formatted(sql: &str, format: Format) -> (u8, String, String) {
        let script = Script::new(sql.to_string());
        captured(format, |printer| run(&script, printer))
    }

    /// Runs `sql` with `result` in place of the SQL engine.
    fn run_with(
        sql: &str,
        mut result: impl FnMut(&str) -> PolarsResult<DataFrame>,
    ) -> (u8, String, String) {
        let script = Script::new(sql.to_string());
        captured(Format::Table, |printer| {
            run_statements(&script, printer, |sql| Ok(result(sql)?.lazy()))
        })
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
    fn a_result_is_written_in_the_chosen_format() {
        let sql = "SELECT n, n * 2 AS twice FROM generate_series(1, 3) AS t (n)";

        let (code, out, err) = run_formatted(sql, Format::Csv);
        assert_eq!((code, err.as_str()), (exit_code::OK, ""));
        assert_eq!(out, "n,twice\n1,2\n2,4\n3,6\n");

        let (_, out, _) = run_formatted(sql, Format::Ndjson);
        assert_eq!(
            out,
            "{\"n\":1,\"twice\":2}\n{\"n\":2,\"twice\":4}\n{\"n\":3,\"twice\":6}\n"
        );
    }

    #[test]
    fn an_empty_result_keeps_its_columns() {
        let (code, out, _) = run_formatted("SELECT 1 AS a, 'x' AS b WHERE false", Format::Csv);
        assert_eq!((code, out.as_str()), (exit_code::OK, "a,b\n"));

        let (_, out, _) = run_captured("SELECT 1 AS a WHERE false");
        assert!(out.starts_with("shape: (0, 1)\n"), "{out}");
    }

    #[test]
    fn a_large_result_arrives_complete_and_in_order() {
        let (code, out, _) = run_formatted(
            "SELECT n FROM generate_series(1, 200000) AS t (n) ORDER BY n DESC",
            Format::Csv,
        );
        assert_eq!(code, exit_code::OK);
        let expected: String = std::iter::once("n".to_string())
            .chain((1..=200_000).rev().map(|n| n.to_string()))
            .map(|line| line + "\n")
            .collect();
        assert!(
            out == expected,
            "the CSV output differs from the expected rows"
        );
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
        let mut printer = Printer::new(&mut out, &mut err, Format::Table);
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
