#[cfg(feature = "csv-encoding")]
use encoding_rs::{Encoding, UTF_8};

use super::*;

/// Write a DataFrame to csv.
///
/// Don't use a `Buffered` writer, the `CsvWriter` internally already buffers writes.
#[must_use]
pub struct CsvWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    options: write_impl::SerializeOptions,
    header: bool,
    batch_size: usize,
    #[cfg(feature = "csv-encoding")]
    encoding: Option<&'static Encoding>,
}

impl<W> SerWriter<W> for CsvWriter<W>
where
    W: Write,
{
    fn new(buffer: W) -> Self {
        // 9f: all nanoseconds
        let options = write_impl::SerializeOptions {
            time_format: Some("%T%.9f".to_string()),
            ..Default::default()
        };

        CsvWriter {
            buffer,
            options,
            header: true,
            batch_size: 1024,
            #[cfg(feature = "csv-encoding")]
            encoding: None,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        match self.encoding {
            Some(enc) => {
                let mut writer = transcoding::TranscodingWriter::new(&mut self.buffer, enc);
                Self::finish_with_writer(
                    &mut writer,
                    df,
                    self.header,
                    self.batch_size,
                    &self.options,
                )
            }
            None => Self::finish_with_writer(
                &mut self.buffer,
                df,
                self.header,
                self.batch_size,
                &self.options,
            ),
        }
    }
}

impl<W> CsvWriter<W>
where
    W: Write,
{
    fn finish_with_writer(
        writer: &mut impl Write,
        df: &mut DataFrame,
        header: bool,
        batch_size: usize,
        options: &write_impl::SerializeOptions,
    ) -> PolarsResult<()> {
        let names = df.get_column_names();
        if header {
            write_impl::write_header(writer, &names, options)?;
        }
        write_impl::write(writer, df, batch_size, options)
    }

    /// Set whether to write headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.header = has_header;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.options.delimiter = delimiter;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    #[cfg(feature = "csv-encoding")]
    /// Set the CSV file's encoding
    pub fn with_encoding(mut self, encoding: Option<String>) -> PolarsResult<Self> {
        // Try to get encoding from given string
        let encoding = encoding.map(|e| {
            Encoding::for_label(e.as_bytes()).ok_or(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("unknown encoding '{}'", e),
            ))
        });
        match encoding {
            Some(Err(err)) => Err(err.into()),
            Some(Ok(enc)) => {
                // If we obtained an encoding, we only want to set the encoding for non-UTF8
                // as any other encoding is much slower.
                self.encoding = match enc {
                    utf8 if utf8 == UTF_8 => None,
                    _ => Some(enc),
                };
                Ok(self)
            }
            None => {
                self.encoding = None;
                Ok(self)
            }
        }
    }

    /// Set the CSV file's date format
    pub fn with_date_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.date_format = format;
        }
        self
    }

    /// Set the CSV file's time format
    pub fn with_time_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.time_format = format;
        }
        self
    }

    /// Set the CSV file's datetime format
    pub fn with_datetime_format(mut self, format: Option<String>) -> Self {
        if format.is_some() {
            self.options.datetime_format = format;
        }
        self
    }

    /// Set the CSV file's float precision
    pub fn with_float_precision(mut self, precision: Option<usize>) -> Self {
        if precision.is_some() {
            self.options.float_precision = precision;
        }
        self
    }

    /// Set the single byte character used for quoting
    pub fn with_quoting_char(mut self, char: u8) -> Self {
        self.options.quote = char;
        self
    }

    /// Set the CSV file's null value representation
    pub fn with_null_value(mut self, null_value: String) -> Self {
        self.options.null = null_value;
        self
    }
}
