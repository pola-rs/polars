use async_trait::async_trait;
use opendal::Operator;
use polars_core::prelude::ArrowSchema;
use polars_error::{to_compute_err, PolarsResult};

use crate::input::files_async::{FileFormat, DEFAULT_SCHEMA_INFER_MAX_RECORD};

#[derive(Debug)]
pub struct CSVFormat {
    schema_infer_max_records: Option<usize>,
    has_header: bool,
    delimiter: u8,
    comment_char: Option<u8>,
    try_parse_dates: bool,
}

impl Default for CSVFormat {
    fn default() -> Self {
        Self {
            schema_infer_max_records: Some(DEFAULT_SCHEMA_INFER_MAX_RECORD),
            has_header: true,
            delimiter: b',',
            comment_char: None,
            try_parse_dates: true,
        }
    }
}

impl CSVFormat {
    /// Construct a new Format with no local overrides
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a limit in terms of records to scan to infer the schema
    /// The default is `DEFAULT_SCHEMA_INFER_MAX_RECORD`
    pub fn set_schema_infer_max_records(mut self, schema_infer_max_records: Option<usize>) -> Self {
        self.schema_infer_max_records = schema_infer_max_records;
        self
    }

    /// Whether to treat the first row as a special header row.
    /// The default is `true`.
    pub fn set_has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Sets the field delimiter to use when parsing CSV.
    ///
    /// The default is `b','`.
    pub fn set_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// The comment character to use when parsing CSV.
    /// If the start of a record begins with the byte given here, then that line is ignored by the CSV parser.
    /// This is disabled by default.
    pub fn set_comment_char(mut self, comment_char: Option<u8>) -> Self {
        self.comment_char = comment_char;
        self
    }

    /// Automatically try to parse dates/ datetimes and time.
    /// If parsing fails, columns remain of dtype `[DataType::Utf8]`.
    /// The default is `true`.
    pub fn set_try_parse_dates(mut self, try_parse_dates: bool) -> Self {
        self.try_parse_dates = try_parse_dates;
        self
    }
}

#[async_trait]
impl FileFormat for CSVFormat {
    /// Read and parse the schema of the CSV file at location `path`
    async fn fetch_schema_async(
        &self,
        store_op: &Operator,
        path: String,
    ) -> PolarsResult<ArrowSchema> {
        let reader = store_op
            .reader(path.as_str())
            .await
            .map_err(to_compute_err)?;

        let mut async_reader = arrow::io::csv::read_async::AsyncReaderBuilder::new()
            .delimiter(self.delimiter)
            .comment(self.comment_char)
            .has_headers(self.has_header)
            .create_reader(reader);

        let (fields, _) = arrow::io::csv::read_async::infer_schema(
            &mut async_reader,
            self.schema_infer_max_records,
            self.has_header,
            &|input: &[u8]| {
                crate::csv::utils::infer_field_schema(
                    std::str::from_utf8(input).unwrap(),
                    self.try_parse_dates,
                )
                .to_arrow()
            },
        )
        .await
        .map_err(to_compute_err)?;

        Ok(ArrowSchema::from(fields))
    }
}
