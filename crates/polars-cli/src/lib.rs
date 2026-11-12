//! Command-line SQL engine for Polars.
//!
//! This crate is unstable: the command line, the output and this library can change in any
//! release.
//!
//! The binary only parses arguments and sets up the printer. Everything that runs SQL goes
//! through [`run`], which reports what happens as [`Event`]s.

use std::io::Write;

use polars::prelude::{DataFrame, Engine, PolarsResult};
use polars::sql::SQLContext;

mod args;
mod error;
mod printer;

pub use args::{HELP, Invocation, UsageError, parse_args};
pub use error::{ErrorRecord, Location};
pub use printer::{Event, Printer};

/// The exit codes of the `polars` command.
pub mod exit_code {
    pub const OK: u8 = 0;
    /// A statement failed, or its result could not be written.
    pub const FAILURE: u8 = 1;
    /// The command line could not be understood.
    pub const USAGE: u8 = 2;
}

/// Runs `sql` and reports the outcome to `printer`. Returns the exit code.
pub fn run<O: Write, E: Write>(sql: &str, printer: &mut Printer<O, E>) -> u8 {
    let mut session = Session::new();
    let outcome = session.run_statement(1, sql, printer);
    drop(session);
    // Spill files are deleted in the background; wait for that before the process can exit.
    polars_ooc::flush_ooc_cleanup();

    match outcome {
        Ok(true) => exit_code::OK,
        // The statement failed, or the output cannot be written to.
        Ok(false) | Err(_) => exit_code::FAILURE,
    }
}

/// The state that statements of one run share.
struct Session {
    context: SQLContext,
}

impl Session {
    fn new() -> Self {
        Self {
            context: SQLContext::new(),
        }
    }

    /// Runs one statement. Returns whether it succeeded; `Err` means the printer failed.
    fn run_statement<O: Write, E: Write>(
        &mut self,
        number: usize,
        sql: &str,
        printer: &mut Printer<O, E>,
    ) -> std::io::Result<bool> {
        match self.execute(sql) {
            Ok(result) => {
                printer.event(Event::Schema(result.schema().clone()))?;
                printer.event(Event::Rows(result))?;
                printer.event(Event::StatementEnd)?;
                Ok(true)
            },
            Err(error) => {
                printer.event(Event::Error(ErrorRecord::from_polars(&error, number)))?;
                Ok(false)
            },
        }
    }

    fn execute(&mut self, sql: &str) -> PolarsResult<DataFrame> {
        let plan = self.context.execute(sql)?;
        Ok(plan.collect_with_engine(Engine::Streaming)?.unwrap_single())
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{DataType, IDX_DTYPE};

    use super::*;

    fn run_captured(sql: &str) -> (u8, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run(sql, &mut Printer::new(&mut out, &mut err));
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
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
    fn rows_before_an_error_are_not_printed() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut printer = Printer::new(&mut out, &mut err);
        let rows = SQLContext::new()
            .execute("SELECT 1 AS a")
            .unwrap()
            .collect()
            .unwrap();

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

    #[test]
    fn failing_output_is_a_failure() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let code = run("SELECT 1 AS a", &mut Printer::new(Closed, Vec::new()));
        assert_eq!(code, exit_code::FAILURE);
    }
}
