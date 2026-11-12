use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use polars::io::csv::write::{BatchedWriter as CsvBatches, CsvWriter};
use polars::io::json::BatchedWriter as NdjsonBatches;
use polars::prelude::{DataFrame, PolarsResult, SchemaRef, SerWriter};

use crate::error::ErrorRecord;

/// How results are written to the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// A table for reading; a long result is cut to its first and last rows
    #[default]
    Table,
    /// Every row, with a header line
    Csv,
    /// Every row, one JSON object per line
    Ndjson,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Table => "table",
            Self::Csv => "csv",
            Self::Ndjson => "ndjson",
        })
    }
}

/// What happens while a script runs.
///
/// A run begins with `Start`. A statement that returns rows sends `Schema`, then any number of
/// `Rows`. Every statement ends with `StatementEnd`, `Error` or `Interrupted`. Nothing follows
/// `Error` or `Interrupted`.
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
///
/// A table is printed when its statement ends, so rows that were sent before a failure never
/// read as a complete result. The other formats are written batch by batch, to keep a large
/// result out of memory; what was written before a failure stays written.
pub struct Printer<O: Write, E> {
    out: Shared<O>,
    err: E,
    format: Format,
    /// The result of the statement that is running.
    result: Option<ResultWriter<O>>,
    /// Whether errors name their statement: only when the script has several.
    number_statements: bool,
}

enum ResultWriter<O: Write> {
    Table(DataFrame),
    Csv(Box<CsvBatches<Shared<O>>>),
    Ndjson(NdjsonBatches<Shared<O>>),
}

impl<O: Write, E: Write> Printer<O, E> {
    pub fn new(out: O, err: E, format: Format) -> Self {
        Self {
            out: Shared {
                writer: Arc::new(Mutex::new(out)),
                closed: Arc::new(AtomicBool::new(false)),
            },
            err,
            format,
            result: None,
            number_statements: false,
        }
    }

    /// Whether the reader of the output has gone away, as when it is piped into `head`.
    pub fn output_is_closed(&self) -> bool {
        self.out.closed.load(Ordering::Relaxed)
    }

    pub fn event(&mut self, event: Event) -> PolarsResult<()> {
        match event {
            Event::Start { statements } => self.number_statements = statements > 1,
            Event::Schema(schema) => self.result = Some(self.start_result(&schema)?),
            Event::Rows(mut rows) => {
                if self.result.is_none() {
                    self.result = Some(self.start_result(rows.schema())?);
                }
                // The writers need the columns of a batch chunked alike.
                rows.align_chunks();
                match self.result.as_mut().unwrap() {
                    ResultWriter::Table(table) => {
                        table.vstack_mut(&rows)?;
                    },
                    ResultWriter::Csv(writer) => writer.write_batch(&rows)?,
                    ResultWriter::Ndjson(writer) => writer.write_batch(&rows)?,
                }
            },
            Event::StatementEnd => {
                match self.result.take() {
                    Some(ResultWriter::Table(table)) => writeln!(self.out, "{table}")?,
                    Some(ResultWriter::Csv(mut writer)) => writer.finish()?,
                    Some(ResultWriter::Ndjson(_)) | None => {},
                }
                self.out.flush()?;
            },
            Event::Error(error) => {
                self.result = None;
                // What was written of the results comes before the error. The output may be
                // what failed, so the error is reported whether or not this works.
                let _ = self.out.flush();
                if self.number_statements {
                    write!(self.err, "statement {}: ", error.statement)?;
                }
                writeln!(self.err, "{}: {}", error.class, error.message)?;
                self.err.flush()?;
            },
            Event::Interrupted => {
                self.result = None;
                let _ = self.out.flush();
                writeln!(self.err, "interrupted")?;
                self.err.flush()?;
            },
        }
        Ok(())
    }

    fn start_result(&self, schema: &SchemaRef) -> PolarsResult<ResultWriter<O>> {
        let out = self.out.clone();
        Ok(match self.format {
            Format::Table => ResultWriter::Table(DataFrame::empty_with_schema(schema)),
            Format::Csv => ResultWriter::Csv(Box::new(CsvWriter::new(out).batched(schema)?)),
            Format::Ndjson => ResultWriter::Ndjson(NdjsonBatches::new(out)),
        })
    }
}

/// The output, shared between the printer and the writer of the result that is in progress.
struct Shared<W> {
    writer: Arc<Mutex<W>>,
    /// Set when a write fails because nobody reads the output any more. Some writers turn
    /// that failure into an error that no longer tells.
    closed: Arc<AtomicBool>,
}

impl<W> Clone for Shared<W> {
    fn clone(&self) -> Self {
        Self {
            writer: Arc::clone(&self.writer),
            closed: Arc::clone(&self.closed),
        }
    }
}

impl<W: Write> Shared<W> {
    fn with<T>(&self, write: impl FnOnce(&mut W) -> io::Result<T>) -> io::Result<T> {
        let result = write(&mut self.writer.lock().unwrap_or_else(PoisonError::into_inner));
        if matches!(&result, Err(error) if error.kind() == io::ErrorKind::BrokenPipe) {
            self.closed.store(true, Ordering::Relaxed);
        }
        result
    }
}

impl<W: Write> Write for Shared<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.with(|writer| writer.write(buffer))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.with(|writer| writer.flush())
    }
}
