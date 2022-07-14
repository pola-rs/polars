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
}

impl<W> SerWriter<W> for CsvWriter<W>
where
    W: Write,
{
    fn new(buffer: W) -> Self {
        // 9f: all nanoseconds
        let options = write_impl::SerializeOptions {
            time_format: Some("%T%.9f".to_string()),
            datetime_format: Some("%FT%H:%M:%S.%9f".to_string()),
            ..Default::default()
        };

        CsvWriter {
            buffer,
            options,
            header: true,
            batch_size: 1024,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> Result<()> {
        df.as_single_chunk_par();
        let names = df.get_column_names();
        if self.header {
            write_impl::write_header(&mut self.buffer, &names, &self.options)?;
        }
        write_impl::write(&mut self.buffer, df, self.batch_size, &self.options)
    }
}

impl<W> CsvWriter<W>
where
    W: Write,
{
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

    /// Set the CSV file's date format
    pub fn with_date_format(mut self, format: Option<String>) -> Self {
        self.options.date_format = format;
        self
    }

    /// Set the CSV file's time format
    pub fn with_time_format(mut self, format: Option<String>) -> Self {
        self.options.time_format = format;
        self
    }

    /// Set the CSV file's timestamp format array in
    pub fn with_datetime(mut self, format: Option<String>) -> Self {
        self.options.datetime_format = format;
        self
    }

    /// Set the single byte character used for quoting
    pub fn with_quoting_char(mut self, char: u8) -> Self {
        self.options.quote = char;
        self
    }
}
