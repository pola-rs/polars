use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use polars::io::utils::file::{Writable, WritableTrait};
use polars::prelude::{
    CompatLevel, CsvWriterOptions, DataFrame, DataType, FileWriteFormat, NDJsonWriterOptions,
    ParquetWriteOptions, PolarsResult, SchemaExt, SchemaRef, SinkTarget, SpecialEq, polars_bail,
    polars_ensure,
};
use polars_arrow::io::ipc::write::{StreamWriter as ArrowStream, WriteOptions};
use unicode_width::UnicodeWidthChar;

use crate::error::{ErrorRecord, LocationKind, SourceLocation};

/// How results are written to the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// A table for reading; a long result is cut to its first and last rows. The result of
    /// DESCRIBE or EXPLAIN is printed in full, as plain text
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
/// A run begins with `Start`. A statement begins with `Statement`. One that returns rows
/// sends `Schema`, then any number of `Rows`. Every statement ends with `StatementEnd`,
/// `Error` or `Interrupted`. Nothing follows `Error` or `Interrupted`.
#[derive(Debug)]
pub enum Event {
    /// The run begins. `statements` is how many statements the script holds.
    Start { statements: usize },
    /// A statement begins. `describes` tells that its result describes a table, a query or a
    /// plan: the table format prints such a result in full.
    Statement { describes: bool },
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
    /// Whether the script has several statements. Errors then name their statement.
    several_statements: bool,
    /// Whether the output is a terminal.
    on_terminal: bool,
    /// Whether the table format prints the result of the current statement in full.
    in_full: bool,
    /// The name errors give the script.
    source: String,
    /// Whether errors show the plan and the context they have.
    verbose: bool,
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
            several_statements: false,
            on_terminal: false,
            in_full: false,
            source: "<command>".to_string(),
            verbose: false,
        }
    }

    pub fn source(mut self, source: String) -> Self {
        self.source = source;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Tells whether the output is a terminal, which does not take a binary format.
    pub fn on_terminal(mut self, on_terminal: bool) -> Self {
        self.on_terminal = on_terminal;
        self
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
        Some((self.sink_target(), format))
    }

    /// For `COPY ... TO STDOUT`: the output as the target of a sink that writes `format`, the
    /// format of the statement. A binary format is one file, which a terminal cannot show and
    /// which cannot share the output with another result.
    pub(crate) fn copy_target(&self, format: &FileWriteFormat) -> PolarsResult<SinkTarget>
    where
        O: Send + 'static,
    {
        if !matches!(format, FileWriteFormat::Csv(_) | FileWriteFormat::NDJson(_)) {
            let name = format.extension();
            polars_ensure!(
                !self.on_terminal,
                InvalidOperation:
                "COPY TO STDOUT format '{name}' is binary; send the output to a file or a pipe"
            );
            polars_ensure!(
                !self.several_statements,
                InvalidOperation:
                "COPY TO STDOUT format '{name}' writes one file, so it takes a script of one statement"
            );
        }
        Ok(self.sink_target())
    }

    fn sink_target(&self) -> SinkTarget
    where
        O: Send + 'static,
    {
        let writer = Writable::Dyn(Box::new(self.out.clone()));
        SinkTarget::Dyn(SpecialEq::new(Arc::new(Mutex::new(Some(writer)))))
    }

    pub fn event(&mut self, event: Event) -> PolarsResult<()> {
        match event {
            Event::Start { statements } => self.several_statements = statements > 1,
            Event::Statement { describes } => self.in_full = describes,
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
                    Some(ResultWriter::Table(table)) if self.in_full => {
                        write_in_full(&mut self.out, &table)?
                    },
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
                write_error(
                    &mut self.err,
                    &error,
                    &self.source,
                    self.several_statements,
                    self.verbose,
                )?;
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

/// Writes `error` as a compiler writes a diagnostic.
fn write_error(
    err: &mut impl Write,
    error: &ErrorRecord,
    source: &str,
    several_statements: bool,
    verbose: bool,
) -> io::Result<()> {
    if several_statements {
        write!(err, "statement {}: ", error.statement)?;
    }
    // The first paragraph heads the report; the others follow the location.
    let (headline, rest) = match error.message.split_once("\n\n") {
        Some((headline, rest)) => (headline, Some(rest)),
        None => (error.message.as_str(), None),
    };
    writeln!(err, "{}: {headline}", error.class)?;
    let digits = error
        .location
        .as_ref()
        .map_or(1, |location| location.start.line.to_string().len());
    let gutter = " ".repeat(digits + 1);
    let exact = matches!(
        error.location,
        Some(SourceLocation {
            kind: LocationKind::Exact,
            ..
        })
    );
    if let Some(location) = &error.location {
        let label = match &location.kind {
            LocationKind::Exact => String::new(),
            LocationKind::Clause(clause) => format!(" (in this {clause} clause)"),
            LocationKind::Statement => " (statement starts here)".to_string(),
        };
        let start = location.start;
        writeln!(
            err,
            "{}--> {source}:{}:{}{label}",
            &gutter[1..],
            start.line,
            start.column
        )?;
        let mut width = match location.end.line == start.line {
            true => location.end.column.saturating_sub(start.column).max(1),
            false => usize::MAX,
        };
        // The span of a function call ends at its last argument: take its parentheses too.
        let chars: Vec<char> = location.line.chars().collect();
        let from = start.column.saturating_sub(1);
        if width != usize::MAX {
            let mut open = chars
                .iter()
                .skip(from)
                .take(width)
                .fold(0, |open, c| match c {
                    '(' => open + 1,
                    ')' => open - 1,
                    _ => open,
                });
            while open > 0 {
                match (from + width..chars.len()).find(|&i| !chars[i].is_whitespace()) {
                    Some(i) if chars[i] == ')' => {
                        width = i + 1 - from;
                        open -= 1;
                    },
                    _ => break,
                }
            }
        }
        let (text, caret, width) = excerpt(&location.line, from, width);
        writeln!(err, "{gutter}|")?;
        writeln!(err, "{} | {text}", start.line)?;
        if exact {
            writeln!(err, "{gutter}| {}{}", " ".repeat(caret), "^".repeat(width))?;
        }
    }
    if let Some(expression) = error.expression.as_ref().filter(|_| !exact) {
        writeln!(err, "{gutter}= expression: {expression}")?;
    }
    for hint in &error.hints {
        writeln!(err, "{gutter}= hint: {hint}")?;
    }
    if let Some(rest) = rest {
        writeln!(err, "\n{}", rest.trim_end())?;
    }
    if verbose && !error.details.is_empty() {
        writeln!(err, "\n{}", error.details.trim_end())?;
    }
    Ok(())
}

const EXCERPT_WIDTH: usize = 100;

/// The shown part of `line` around the `width` characters at `at`, and their terminal columns.
fn excerpt(line: &str, at: usize, width: usize) -> (String, usize, usize) {
    // One space per tab or other control character keeps the underline aligned.
    let chars: Vec<char> = line
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let at = at.min(chars.len());
    let (start, end) = match chars.len() <= EXCERPT_WIDTH {
        true => (0, chars.len()),
        false => {
            let start = at
                .saturating_sub(EXCERPT_WIDTH / 2)
                .min(chars.len() - EXCERPT_WIDTH);
            (start, start + EXCERPT_WIDTH)
        },
    };
    let cells = |chars: &[char]| -> usize { chars.iter().map(|c| c.width().unwrap_or(0)).sum() };
    let mut text = String::new();
    let mut caret = cells(&chars[start..at]);
    if start > 0 {
        text.push_str("...");
        caret += 3;
    }
    text.extend(&chars[start..end]);
    if end < chars.len() {
        text.push_str("...");
    }
    let shown = (at + width.min(end.saturating_sub(at))).min(end);
    (text, caret, cells(&chars[at..shown]).max(1))
}

/// Writes every row of `table` as plain text, where the table format cuts rows and long
/// values. One column is written as its values, one per line. Several columns get a header
/// line and are padded to line up.
fn write_in_full(out: &mut impl Write, table: &DataFrame) -> PolarsResult<()> {
    let mut columns = Vec::with_capacity(table.width());
    for column in table.columns() {
        let text = column.cast(&DataType::String)?;
        let mut cells = Vec::with_capacity(table.height() + 1);
        if table.width() > 1 {
            cells.push(column.name().to_string());
        }
        cells.extend(text.str()?.iter().map(|v| v.unwrap_or("null").to_string()));
        let width = cells.iter().map(|cell| cell.chars().count()).max();
        columns.push((cells, width.unwrap_or(0)));
    }

    let lines = columns.first().map_or(0, |(cells, _)| cells.len());
    for line in 0..lines {
        for (index, (cells, width)) in columns.iter().enumerate() {
            match index + 1 == columns.len() {
                // Nothing follows the last column, so it needs no padding.
                true => writeln!(out, "{}", cells[line])?,
                false => write!(out, "{:width$}  ", cells[line])?,
            }
        }
    }
    Ok(())
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
