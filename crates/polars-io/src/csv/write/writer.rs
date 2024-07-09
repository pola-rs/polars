use std::io::Write;
use std::num::NonZeroUsize;

use polars_core::frame::DataFrame;
use polars_core::schema::{IndexOfSchema, Schema};
use polars_core::POOL;
use polars_error::PolarsResult;

use super::write_impl::{write, write_bom, write_header};
use super::{QuoteStyle, SerializeOptions};
use crate::shared::SerWriter;

/// Write a DataFrame to csv.
///
/// Don't use a `Buffered` writer, the `CsvWriter` internally already buffers writes.
#[must_use]
pub struct CsvWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    options: SerializeOptions,
    header: bool,
    bom: bool,
    batch_size: NonZeroUsize,
    n_threads: usize,
}

impl<W> SerWriter<W> for CsvWriter<W>
where
    W: Write,
{
    fn new(buffer: W) -> Self {
        // 9f: all nanoseconds
        let options = SerializeOptions {
            time_format: Some("%T%.9f".to_string()),
            ..Default::default()
        };

        CsvWriter {
            buffer,
            options,
            header: true,
            bom: false,
            batch_size: NonZeroUsize::new(1024).unwrap(),
            n_threads: POOL.current_num_threads(),
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        if self.bom {
            write_bom(&mut self.buffer)?;
        }
        let names = df.get_column_names();
        if self.header {
            write_header(&mut self.buffer, &names, &self.options)?;
        }
        write(
            &mut self.buffer,
            df,
            self.batch_size.into(),
            &self.options,
            self.n_threads,
        )
    }
}

impl<W> CsvWriter<W>
where
    W: Write,
{
    /// Set whether to write UTF-8 BOM.
    pub fn include_bom(mut self, include_bom: bool) -> Self {
        self.bom = include_bom;
        self
    }

    /// Set whether to write headers.
    pub fn include_header(mut self, include_header: bool) -> Self {
        self.header = include_header;
        self
    }

    /// Set the CSV file's column separator as a byte character.
    pub fn with_separator(mut self, separator: u8) -> Self {
        self.options.separator = separator;
        self
    }

    /// Set the batch size to use while writing the CSV.
    pub fn with_batch_size(mut self, batch_size: NonZeroUsize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the CSV file's date format.
    pub fn with_date_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.date_format = format;
        }
        self
    }

    /// Set the CSV file's time format.
    pub fn with_time_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.time_format = format;
        }
        self
    }

    /// Set the CSV file's datetime format.
    pub fn with_datetime_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.datetime_format = format;
        }
        self
    }

    /// Set the CSV file's forced scientific notation for floats.
    pub fn with_float_scientific(mut self, scientific: Option<bool>) -> Self {
        if scientific.is_some() {
            self.options.float_scientific = scientific;
        }
        self
    }

    /// Set the CSV file's float precision.
    pub fn with_float_precision(mut self, precision: Option<usize>) -> Self {
        if precision.is_some() {
            self.options.float_precision = precision;
        }
        self
    }

    /// Set the single byte character used for quoting.
    pub fn with_quote_char(mut self, char: u8) -> Self {
        self.options.quote_char = char;
        self
    }

    /// Set the CSV file's null value representation.
    pub fn with_null_value(mut self, null_value: String) -> Self {
        self.options.null = null_value;
        self
    }

    /// Set the CSV file's line terminator.
    pub fn with_line_terminator(mut self, line_terminator: String) -> Self {
        self.options.line_terminator = line_terminator;
        self
    }

    /// Set the CSV file's quoting behavior.
    /// See more on [`QuoteStyle`].
    pub fn with_quote_style(mut self, quote_style: QuoteStyle) -> Self {
        self.options.quote_style = quote_style;
        self
    }

    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let expects_bom = self.bom;
        let expects_header = self.header;
        Ok(BatchedWriter {
            writer: self,
            has_written_bom: !expects_bom,
            has_written_header: !expects_header,
            schema: schema.clone(),
        })
    }
}

pub struct BatchedWriter<W: Write> {
    writer: CsvWriter<W>,
    has_written_bom: bool,
    has_written_header: bool,
    schema: Schema,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the csv writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        if !self.has_written_bom {
            self.has_written_bom = true;
            write_bom(&mut self.writer.buffer)?;
        }

        if !self.has_written_header {
            self.has_written_header = true;
            let names = df.get_column_names();
            write_header(&mut self.writer.buffer, &names, &self.writer.options)?;
        }

        write(
            &mut self.writer.buffer,
            df,
            self.writer.batch_size.into(),
            &self.writer.options,
            self.writer.n_threads,
        )?;
        Ok(())
    }

    /// Writes the header of the csv file if not done already. Returns the total size of the file.
    pub fn finish(&mut self) -> PolarsResult<()> {
        if !self.has_written_bom {
            self.has_written_bom = true;
            write_bom(&mut self.writer.buffer)?;
        }

        if !self.has_written_header {
            self.has_written_header = true;
            let names = self.schema.get_names();
            write_header(&mut self.writer.buffer, &names, &self.writer.options)?;
        };

        Ok(())
    }
}
