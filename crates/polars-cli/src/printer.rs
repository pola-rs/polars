use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use polars::io::utils::file::{Writable, WritableTrait};
use polars::prelude::{
    CompatLevel, CsvWriterOptions, DataFrame, FileWriteFormat, NDJsonWriterOptions,
    ParquetWriteOptions, PolarsResult, SchemaExt, SchemaRef, SinkTarget, SpecialEq, polars_bail,
};
use polars_arrow::io::ipc::write::{StreamWriter as ArrowStream, WriteOptions};

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
    /// Every row, as one Parquet file; for a script of one statement
    Parquet,
    /// Every row, in the Arrow IPC stream format; for a script of one statement
    ArrowStream,
}

impl Format {
    /// Whether the output is one binary file, which cannot hold a second result.
    pub fn is_one_file(self) -> bool {
        matches!(self, Self::Parquet | Self::ArrowStream)
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Table => "table",
            Self::Csv => "csv",
            Self::Ndjson => "ndjson",
            Self::Parquet => "parquet",
            Self::ArrowStream => "arrow-stream",
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
/// read as a complete result. The other formats are written while the statement runs, to keep
/// a large result out of memory; what was written before a failure stays written. Where the
/// engine has a sink for the format, the engine writes it: see [`Printer::engine_sink`].
pub struct Printer<O: Write, E> {
    out: Shared<O>,
    err: E,
    format: Format,
    result: Option<ResultWriter<O>>,
    /// Whether errors name their statement: only when the script has several.
    number_statements: bool,
}

enum ResultWriter<O: Write> {
    Table(DataFrame),
    ArrowStream(Box<ArrowStream<Shared<O>>>),
}

/// The Arrow types that older readers know, as Polars' own stream writer uses.
const ARROW_COMPAT: CompatLevel = CompatLevel::oldest();

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

    /// For a format that the engine writes itself: the format, and the output as the target of
    /// a sink. The output is then what `sink_csv`, `sink_ndjson` or `sink_parquet` writes. The
    /// engine has no sink for the Arrow IPC stream format.
    pub(crate) fn engine_sink(&self) -> Option<(SinkTarget, FileWriteFormat)>
    where
        O: Send + 'static,
    {
        let format = match self.format {
            Format::Csv => FileWriteFormat::Csv(CsvWriterOptions::default()),
            Format::Ndjson => FileWriteFormat::NDJson(NDJsonWriterOptions::default()),
            Format::Parquet => FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions::default())),
            Format::Table | Format::ArrowStream => return None,
        };
        let writer = Writable::Dyn(Box::new(self.out.clone()));
        let target = SinkTarget::Dyn(SpecialEq::new(Arc::new(Mutex::new(Some(writer)))));
        Some((target, format))
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
                    ResultWriter::ArrowStream(writer) => {
                        for batch in rows.iter_chunks(ARROW_COMPAT, true) {
                            writer.write(&batch, None)?;
                        }
                    },
                }
            },
            Event::StatementEnd => {
                match self.result.take() {
                    Some(ResultWriter::Table(table)) => writeln!(self.out, "{table}")?,
                    Some(ResultWriter::ArrowStream(mut writer)) => writer.finish()?,
                    None => {},
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
            Format::ArrowStream => {
                let mut writer = ArrowStream::new(out, WriteOptions { compression: None });
                writer.start(&schema.to_arrow(ARROW_COMPAT), None)?;
                ResultWriter::ArrowStream(Box::new(writer))
            },
            Format::Csv | Format::Ndjson | Format::Parquet => {
                polars_bail!(InvalidOperation: "the engine writes format '{}'", self.format)
            },
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

/// The engine closes the file it wrote. The output stays open for the printer.
impl<W: Write> WritableTrait for Shared<W> {
    fn close(&mut self) -> io::Result<()> {
        self.flush()
    }

    fn sync_all(&self) -> io::Result<()> {
        Ok(())
    }

    fn sync_data(&self) -> io::Result<()> {
        Ok(())
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
