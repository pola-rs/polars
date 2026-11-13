use std::io::{self, Write};

use polars::prelude::{DataFrame, SchemaRef};

use crate::error::ErrorRecord;

/// What happens while a script runs.
///
/// A run begins with `Start`. A statement that returns rows sends `Schema`, then any number of
/// `Rows`. Every statement ends with `StatementEnd`, `Error` or `Interrupted`, so rows that were
/// sent before a failure never read as a complete result. Nothing follows `Error` or
/// `Interrupted`.
#[derive(Debug)]
pub enum Event {
    /// The run begins. `statements` is how many statements the script holds.
    Start { statements: usize },
    /// The columns of the result.
    Schema(SchemaRef),
    /// A batch of result rows.
    Rows(DataFrame),
    /// The statement succeeded.
    StatementEnd,
    /// The statement failed.
    Error(ErrorRecord),
    /// The run was interrupted.
    Interrupted,
}

/// Renders events: results to `out`, errors to `err`.
pub struct Printer<O, E> {
    out: O,
    err: E,
    /// The rows of the statement that is running.
    pending: Option<DataFrame>,
    /// Whether errors name their statement: only when the script has several.
    number_statements: bool,
}

impl<O: Write, E: Write> Printer<O, E> {
    pub fn new(out: O, err: E) -> Self {
        Self {
            out,
            err,
            pending: None,
            number_statements: false,
        }
    }

    pub fn event(&mut self, event: Event) -> io::Result<()> {
        match event {
            Event::Start { statements } => self.number_statements = statements > 1,
            Event::Schema(schema) => {
                self.pending = Some(DataFrame::empty_with_schema(&schema));
            },
            Event::Rows(rows) => match &mut self.pending {
                Some(pending) => {
                    pending.vstack_mut(&rows).map_err(io::Error::other)?;
                },
                None => self.pending = Some(rows),
            },
            Event::StatementEnd => {
                if let Some(result) = self.pending.take() {
                    writeln!(self.out, "{result}")?;
                }
                self.out.flush()?;
            },
            Event::Error(error) => {
                self.pending = None;
                // Results of earlier statements come before the error.
                self.out.flush()?;
                if self.number_statements {
                    write!(self.err, "statement {}: ", error.statement)?;
                }
                writeln!(self.err, "{}: {}", error.class, error.message)?;
                self.err.flush()?;
            },
            Event::Interrupted => {
                self.pending = None;
                self.out.flush()?;
                writeln!(self.err, "interrupted")?;
                self.err.flush()?;
            },
        }
        Ok(())
    }
}
