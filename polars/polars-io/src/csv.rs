//! # (De)serializing CSV files
//!
//! ## Maximal performance
//! Currently [CsvReader::new](CsvReader::new) has an extra copy. If you want optimal performance in CSV parsing/
//! reading, it is advised to use [CsvReader::from_path](CsvReader::from_path).
//!
//! ## Write a DataFrame to a csv file.
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example(df: &mut DataFrame) -> Result<()> {
//!     let mut file = File::create("example.csv").expect("could not create file");
//!
//!     CsvWriter::new(&mut file)
//!     .has_header(true)
//!     .with_delimiter(b',')
//!     .finish(df)
//! }
//! ```
//!
//! ## Read a csv file to a DataFrame
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> Result<DataFrame> {
//!     // always prefer `from_path` as that is fastest.
//!     CsvReader::from_path("iris_csv")?
//!             .has_header(true)
//!             .finish()
//! }
//! ```
//!
use crate::aggregations::ScanAggregation;
use crate::csv_core::csv::{cast_columns, CoreReader};
use crate::csv_core::utils::get_reader_bytes;
use crate::mmap::MmapBytesReader;
use crate::predicates::PhysicalIoExpr;
use crate::utils::resolve_homedir;
use crate::{RowCount, SerReader, SerWriter};
pub use arrow::io::csv::write;
use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use rayon::prelude::*;
#[cfg(feature = "temporal")]
use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Write a DataFrame to csv.
#[must_use]
pub struct CsvWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    /// arrow specific options
    options: write::SerializeOptions,
    header: bool,
}

impl<W> SerWriter<W> for CsvWriter<W>
where
    W: Write,
{
    fn new(buffer: W) -> Self {
        // 9f: all nanoseconds
        let options = write::SerializeOptions {
            time64_format: Some("%T%.9f".to_string()),
            timestamp_format: Some("%FT%H:%M:%S.%9f".to_string()),
            ..Default::default()
        };

        CsvWriter {
            buffer,
            options,
            header: true,
        }
    }

    fn finish(mut self, df: &mut DataFrame) -> Result<()> {
        df.rechunk();
        let names = df.get_column_names();
        let iter = df.iter_chunks();
        if self.header {
            write::write_header(&mut self.buffer, &names, &self.options)?;
        }
        for batch in iter {
            write::write_chunk(&mut self.buffer, &batch, &self.options)?;
        }
        Ok(())
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

    /// Set the CSV file's date format
    pub fn with_date_format(mut self, format: Option<String>) -> Self {
        self.options.date32_format = format;
        self
    }

    /// Set the CSV file's time format
    pub fn with_time_format(mut self, format: Option<String>) -> Self {
        self.options.time32_format = format.clone();
        self.options.time64_format = format;
        self
    }

    /// Set the CSV file's timestamp format array in
    pub fn with_timestamp_format(mut self, format: Option<String>) -> Self {
        self.options.timestamp_format = format;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CsvEncoding {
    /// Utf8 encoding
    Utf8,
    /// Utf8 encoding and unknown bytes are replaced with �
    LossyUtf8,
}

#[derive(Clone, Debug)]
pub enum NullValues {
    /// A single value that's used for all columns
    AllColumns(String),
    /// A different null value per column
    Columns(Vec<String>),
    /// Tuples that map column names to null value of that column
    Named(Vec<(String, String)>),
}

impl NullValues {
    /// Use the schema and the null values to produce a null value for every column.
    pub(crate) fn process(self, schema: &Schema) -> Result<Vec<String>> {
        let out = match self {
            NullValues::Columns(v) => v,
            NullValues::AllColumns(v) => (0..schema.len()).map(|_| v.clone()).collect(),
            NullValues::Named(v) => {
                let mut null_values = vec!["".to_string(); schema.len()];
                for (name, null_value) in v {
                    let i = schema.try_index_of(&name)?;
                    null_values[i] = null_value;
                }
                null_values
            }
        };
        Ok(out)
    }
}

/// Create a new DataFrame by reading a csv file.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::prelude::*;
/// use std::fs::File;
///
/// fn example() -> Result<DataFrame> {
///     CsvReader::from_path("iris_csv")?
///             .has_header(true)
///             .finish()
/// }
/// ```
#[must_use]
pub struct CsvReader<'a, R>
where
    R: MmapBytesReader,
{
    /// File or Stream object
    reader: R,
    /// Aggregates chunk afterwards to a single chunk.
    rechunk: bool,
    /// Stop reading from the csv after this number of rows is reached
    n_rows: Option<usize>,
    // used by error ignore logic
    max_records: Option<usize>,
    skip_rows: usize,
    /// Optional indexes of the columns to project
    projection: Option<Vec<usize>>,
    /// Optional column names to project/ select.
    columns: Option<Vec<String>>,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_parser_errors: bool,
    schema: Option<&'a Schema>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<PathBuf>,
    schema_overwrite: Option<&'a Schema>,
    dtype_overwrite: Option<&'a [DataType]>,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    comment_char: Option<u8>,
    null_values: Option<NullValues>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&'a [ScanAggregation]>,
    quote_char: Option<u8>,
    skip_rows_after_header: usize,
    #[cfg(feature = "temporal")]
    parse_dates: bool,
    row_count: Option<RowCount>,
}

impl<'a, R> CsvReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    /// Skip these rows after the header
    pub fn with_skip_rows_after_header(mut self, offset: usize) -> Self {
        self.skip_rows_after_header = offset;
        self
    }

    /// Add a `row_count` column.
    pub fn with_row_count(mut self, rc: Option<RowCount>) -> Self {
        self.row_count = rc;
        self
    }

    /// Sets the chunk size used by the parser. This influences performance
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set  [`CsvEncoding`]
    pub fn with_encoding(mut self, enc: CsvEncoding) -> Self {
        self.encoding = enc;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    pub fn with_ignore_parser_errors(mut self, ignore: bool) -> Self {
        self.ignore_parser_errors = ignore;
        self
    }

    /// Set the CSV file's schema. This only accepts datatypes that are implemented
    /// in the csv parser and expects a complete Schema.
    ///
    /// It is recommended to use [with_dtypes](Self::with_dtypes) instead.
    pub fn with_schema(mut self, schema: &'a Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Skip the first `n` rows during parsing. The header will be parsed an `n` lines.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Rechunk the DataFrame to contiguous memory after the CSV is parsed.
    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Set whether the CSV file has headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Set the comment character. Lines starting with this character will be ignored.
    pub fn with_comment_char(mut self, comment_char: Option<u8>) -> Self {
        self.comment_char = comment_char;
        self
    }

    /// Set values that will be interpreted as missing/ null. Note that any value you set as null value
    /// will not be escaped, so if quotation marks are part of the null value you should include them.
    pub fn with_null_values(mut self, null_values: Option<NullValues>) -> Self {
        self.null_values = null_values;
        self
    }

    /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
    /// of the total schema.
    pub fn with_dtypes(mut self, schema: Option<&'a Schema>) -> Self {
        self.schema_overwrite = schema;
        self
    }

    /// Overwrite the dtypes in the schema in the order of the slice that's given.
    /// This is useful if you don't know the column names beforehand
    pub fn with_dtypes_slice(mut self, dtypes: Option<&'a [DataType]>) -> Self {
        self.dtype_overwrite = dtypes;
        self
    }

    /// Set the CSV reader to infer the schema of the file
    ///
    /// # Arguments
    /// * `max_records` - Maximum number of rows read for schema inference.
    ///                   Setting this to `None` will do a full table scan (slow).
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        // used by error ignore logic
        self.max_records = max_records;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Set the number of threads used in CSV reading. The default uses the number of cores of
    /// your cpu.
    ///
    /// Note that this only works if this is initialized with `CsvReader::from_path`.
    /// Note that the number of cores is the maximum allowed number of threads.
    pub fn with_n_threads(mut self, n: Option<usize>) -> Self {
        self.n_threads = n;
        self
    }

    /// The preferred way to initialize this builder. This allows the CSV file to be memory mapped
    /// and thereby greatly increases parsing performance.
    pub fn with_path<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.path = path.map(|p| p.into());
        self
    }

    /// Sets the size of the sample taken from the CSV file. The sample is used to get statistic about
    /// the file. These statistics are used to try to optimally allocate up front. Increasing this may
    /// improve performance.
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Reduce memory consumption at the expense of performance
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    /// Set the `char` used as quote char. The default is `b'"'`. If set to `[None]` quoting is disabled.
    pub fn with_quote_char(mut self, quote: Option<u8>) -> Self {
        self.quote_char = quote;
        self
    }

    /// Automatically try to parse dates/ datetimes and time. If parsing fails, columns remain of dtype `[DataType::Utf8]`.
    #[cfg(feature = "temporal")]
    pub fn with_parse_dates(mut self, toggle: bool) -> Self {
        self.parse_dates = toggle;
        self
    }

    #[cfg(feature = "private")]
    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }

    pub fn with_aggregate(mut self, aggregate: Option<&'a [ScanAggregation]>) -> Self {
        self.aggregate = aggregate;
        self
    }
}

impl<'a> CsvReader<'a, File> {
    /// This is the recommended way to create a csv reader as this allows for fastest parsing.
    pub fn from_path<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = resolve_homedir(&path.into());
        let f = std::fs::File::open(&path)?;
        Ok(Self::new(f).with_path(Some(path)))
    }
}

impl<'a, R> SerReader<R> for CsvReader<'a, R>
where
    R: MmapBytesReader,
{
    /// Create a new CsvReader from a file/ stream
    fn new(reader: R) -> Self {
        CsvReader {
            reader,
            rechunk: true,
            n_rows: None,
            max_records: Some(128),
            skip_rows: 0,
            projection: None,
            delimiter: None,
            has_header: true,
            ignore_parser_errors: false,
            schema: None,
            columns: None,
            encoding: CsvEncoding::Utf8,
            n_threads: None,
            path: None,
            schema_overwrite: None,
            dtype_overwrite: None,
            sample_size: 1024,
            chunk_size: 1 << 18,
            low_memory: false,
            comment_char: None,
            null_values: None,
            predicate: None,
            aggregate: None,
            quote_char: Some(b'"'),
            skip_rows_after_header: 0,
            #[cfg(feature = "temporal")]
            parse_dates: false,
            row_count: None,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        // we cannot append categorical under local string cache, so we cast them later.
        #[allow(unused_mut)]
        let mut to_cast_local = vec![];

        let mut df = if let Some(schema) = self.schema_overwrite {
            // This branch we check if there are dtypes we cannot parse.
            // We only support a few dtypes in the parser and later cast to the required dtype
            let mut to_cast = Vec::with_capacity(schema.len());

            #[allow(clippy::unnecessary_filter_map)]
            let fields: Vec<_> = schema
                .iter_fields()
                .filter_map(|fld| {
                    use DataType::*;
                    match fld.data_type() {
                        // For categorical we first read as utf8 and later cast to categorical
                        #[cfg(feature = "dtype-categorical")]
                        Categorical(_) => {
                            to_cast_local.push(fld.clone());
                            Some(Field::new(fld.name(), DataType::Utf8))
                        }
                        Date | Datetime(_, _) => {
                            to_cast.push(fld);
                            // let inference decide the column type
                            None
                        }
                        Time => {
                            to_cast.push(fld);
                            // let inference decide the column type
                            None
                        }
                        Int8 | Int16 | UInt8 | UInt16 | Boolean => {
                            // We have not compiled these buffers, so we cast them later.
                            to_cast.push(fld);
                            // let inference decide the column type
                            None
                        }
                        _ => Some(fld),
                    }
                })
                .collect();
            let schema = Schema::from(fields);

            // we cannot overwrite self, because the lifetime is already instantiated with `a, and
            // the lifetime that accompanies this scope is shorter.
            // So we just build_csv_reader from here
            let reader_bytes = get_reader_bytes(&mut self.reader)?;
            let mut csv_reader = CoreReader::new(
                reader_bytes,
                self.n_rows,
                self.skip_rows,
                self.projection,
                self.max_records,
                self.delimiter,
                self.has_header,
                self.ignore_parser_errors,
                self.schema,
                self.columns,
                self.encoding,
                self.n_threads,
                Some(&schema),
                self.dtype_overwrite,
                self.sample_size,
                self.chunk_size,
                self.low_memory,
                self.comment_char,
                self.quote_char,
                self.null_values,
                self.predicate,
                self.aggregate,
                &to_cast,
                self.skip_rows_after_header,
                self.row_count,
            )?;
            csv_reader.as_df()?
        } else {
            let reader_bytes = get_reader_bytes(&mut self.reader)?;
            let mut csv_reader = CoreReader::new(
                reader_bytes,
                self.n_rows,
                self.skip_rows,
                self.projection,
                self.max_records,
                self.delimiter,
                self.has_header,
                self.ignore_parser_errors,
                self.schema,
                self.columns,
                self.encoding,
                self.n_threads,
                self.schema,
                self.dtype_overwrite,
                self.sample_size,
                self.chunk_size,
                self.low_memory,
                self.comment_char,
                self.quote_char,
                self.null_values,
                self.predicate,
                self.aggregate,
                &[],
                self.skip_rows_after_header,
                self.row_count,
            )?;
            csv_reader.as_df()?
        };

        // Important that this rechunk is never done in parallel.
        // As that leads to great memory overhead.
        if rechunk && df.n_chunks()? > 1 {
            if self.low_memory {
                df.as_single_chunk();
            } else {
                df.as_single_chunk_par();
            }
        }
        #[cfg(feature = "temporal")]
        if self.parse_dates {
            // determine the schema that's given by the user. That should not be changed
            let fixed_schema = match (self.schema_overwrite, self.dtype_overwrite) {
                (Some(schema), _) => Cow::Borrowed(schema),
                (None, Some(dtypes)) => {
                    let fields: Vec<_> = dtypes
                        .iter()
                        .zip(df.get_column_names())
                        .map(|(dtype, name)| Field::new(name, dtype.clone()))
                        .collect();

                    Cow::Owned(Schema::from(fields))
                }
                _ => Cow::Owned(Schema::default()),
            };
            df = parse_dates(df, &*fixed_schema)
        }

        cast_columns(&mut df, &to_cast_local, true)?;
        Ok(df)
    }
}

#[cfg(feature = "temporal")]
fn parse_dates(df: DataFrame, fixed_schema: &Schema) -> DataFrame {
    let cols = df
        .get_columns()
        .par_iter()
        .map(|s| {
            if let Ok(ca) = s.utf8() {
                // don't change columns that are in the fixed schema.
                if fixed_schema.index_of(s.name()).is_some() {
                    return s.clone();
                }

                #[cfg(feature = "dtype-time")]
                if let Ok(ca) = ca.as_time(None) {
                    return ca.into_series();
                }
                if let Ok(ca) = ca.as_date(None) {
                    ca.into_series()
                } else if let Ok(ca) = ca.as_datetime(None, TimeUnit::Milliseconds) {
                    ca.into_series()
                } else {
                    s.clone()
                }
            } else {
                s.clone()
            }
        })
        .collect::<Vec<_>>();

    DataFrame::new_no_checks(cols)
}
