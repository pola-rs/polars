use std::io::{self, Write};

use polars::prelude::{DataFrame, SchemaRef};

use crate::error::ErrorRecord;

/// What happens while a statement runs.
///
/// A statement that returns rows sends `Schema`, then any number of `Rows`. Every statement ends
/// with either `StatementEnd` or `Error`, so rows that were sent before a failure never read as a
/// complete result.
#[derive(Debug)]
pub enum Event {
    /// The columns of the result.
    Schema(SchemaRef),
    /// A batch of result rows.
    Rows(DataFrame),
    /// The statement succeeded.
    StatementEnd,
    /// The statement failed.
    Error(ErrorRecord),
}

/// Renders events: results to `out`, errors to `err`.
pub struct Printer<O, E> {
    out: O,
    err: E,
    /// The rows of the statement that is running.
    pending: Option<DataFrame>,
}

impl<O: Write, E: Write> Printer<O, E> {
    pub fn new(out: O, err: E) -> Self {
        Self {
            out,
            err,
            pending: None,
        }
    }

    pub fn event(&mut self, event: Event) -> io::Result<()> {
        match event {
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
                writeln!(self.err, "{}: {}", error.class, error.message)?;
                self.err.flush()?;
            },
        }
        Ok(())
    }
}
