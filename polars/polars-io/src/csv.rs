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
//!             .infer_schema(None)
//!             .has_header(true)
//!             .finish()
//! }
//! ```
//!
use crate::csv_core::csv::CoreReader;
use crate::csv_core::utils::get_reader_bytes;
use crate::mmap::MmapBytesReader;
use crate::utils::resolve_homedir;
use crate::{PhysicalIoExpr, ScanAggregation, SerReader, SerWriter};
pub use arrow::io::csv::write;
use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Write a DataFrame to csv.
pub struct CsvWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    /// Builds an Arrow CSV Writer
    writer_builder: write::WriterBuilder,
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
            writer_builder: write::WriterBuilder::new(),
            options,
            header: true,
        }
    }

    fn finish(self, df: &DataFrame) -> Result<()> {
        let mut writer = self.writer_builder.from_writer(self.buffer);
        let iter = df.iter_record_batches();
        if self.header {
            write::write_header(&mut writer, &df.schema().to_arrow())?;
        }
        for batch in iter {
            write::write_batch(&mut writer, &batch, &self.options)?;
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
        self.writer_builder.delimiter(delimiter);
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

#[derive(Copy, Clone)]
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
                    let i = schema.index_of(&name)?;
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
///             .infer_schema(None)
///             .has_header(true)
///             .finish()
/// }
/// ```
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
    #[cfg(feature = "temporal")]
    parse_dates: bool,
}

impl<'a, R> CsvReader<'a, R>
where
    R: 'a + MmapBytesReader,
{
    /// Sets the chunk size used by the parser. This influences performance
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets the CsvEncoding
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

    /// Skip the first `n` rows during parsing.
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
            chunk_size: 1 << 16,
            low_memory: false,
            comment_char: None,
            null_values: None,
            predicate: None,
            aggregate: None,
            quote_char: Some(b'"'),
            #[cfg(feature = "temporal")]
            parse_dates: false,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;

        let mut df = if let Some(schema) = self.schema_overwrite {
            // This branch we check if there are dtypes we cannot parse.
            // We only support a few dtypes in the parser and later cast to the required dtype
            let mut to_cast = Vec::with_capacity(schema.len());

            let fields = schema
                .fields()
                .iter()
                .filter_map(|fld| {
                    use DataType::*;
                    match fld.data_type() {
                        // For categorical we first read as utf8 and later cast to categorical
                        Categorical => {
                            to_cast.push(fld);
                            Some(Field::new(fld.name(), DataType::Utf8))
                        }
                        Date | Datetime => {
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
                        _ => Some(fld.clone()),
                    }
                })
                .collect();
            let schema = Schema::new(fields);

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
            )?;
            let mut df = csv_reader.as_df()?;

            // cast to the original dtypes in the schema
            for fld in to_cast {
                df.may_apply(fld.name(), |s| s.cast(fld.data_type()))?;
            }
            df
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
                    let fields = dtypes
                        .iter()
                        .zip(df.get_column_names())
                        .map(|(dtype, name)| Field::new(name, dtype.clone()))
                        .collect();

                    Cow::Owned(Schema::new(fields))
                }
                _ => Cow::Owned(Schema::new(vec![])),
            };
            df = parse_dates(df, &*fixed_schema)
        }
        Ok(df)
    }
}

#[cfg(feature = "temporal")]
fn parse_dates(df: DataFrame, fixed_schema: &Schema) -> DataFrame {
    let mut cols: Vec<Series> = df.into();

    for s in cols.iter_mut() {
        if let Ok(ca) = s.utf8() {
            // don't change columns that are in the fixed schema.
            if fixed_schema.column_with_name(s.name()).is_some() {
                continue;
            }

            #[cfg(feature = "dtype-time")]
            if let Ok(ca) = ca.as_time(None) {
                *s = ca.into_series();
                continue;
            }
            // the order is important. A datetime can always be parsed as date.
            if let Ok(ca) = ca.as_datetime(None) {
                *s = ca.into_series()
            } else if let Ok(ca) = ca.as_date(None) {
                *s = ca.into_series()
            }
        }
    }

    DataFrame::new_no_checks(cols)
}

#[cfg(test)]
mod test {
    use crate::csv_core::utils::get_file_chunks;
    use crate::prelude::*;
    use polars_core::datatypes::AnyValue;
    use polars_core::prelude::*;
    use std::io::Cursor;

    #[test]
    fn write_csv() {
        let mut buf: Vec<u8> = Vec::new();
        let df = create_df();

        CsvWriter::new(&mut buf)
            .has_header(true)
            .finish(&df)
            .expect("csv written");
        let csv = std::str::from_utf8(&buf).unwrap();
        assert_eq!("days,temp\n0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);

        let mut buf: Vec<u8> = Vec::new();
        CsvWriter::new(&mut buf)
            .has_header(false)
            .finish(&df)
            .expect("csv written");
        let csv = std::str::from_utf8(&buf).unwrap();
        assert_eq!("0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);
    }

    #[test]
    fn test_read_csv_file() {
        let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
        let file = std::fs::File::open(path).unwrap();
        let df = CsvReader::new(file)
            .with_path(Some(path.to_string()))
            .finish()
            .unwrap();
        dbg!(df);
    }

    #[test]
    fn test_parser() {
        let s = r#"
 "sepal.length","sepal.width","petal.length","petal.width","variety"
 5.1,3.5,1.4,.2,"Setosa"
 4.9,3,1.4,.2,"Setosa"
 4.7,3.2,1.3,.2,"Setosa"
 4.6,3.1,1.5,.2,"Setosa"
 5,3.6,1.4,.2,"Setosa"
 5.4,3.9,1.7,.4,"Setosa"
 4.6,3.4,1.4,.3,"Setosa"
"#;

        let file = Cursor::new(s);
        CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .with_ignore_parser_errors(true)
            .finish()
            .unwrap();

        let s = r#"
         "sepal.length","sepal.width","petal.length","petal.width","variety"
         5.1,3.5,1.4,.2,"Setosa"
         5.1,3.5,1.4,.2,"Setosa"
 "#;

        let file = Cursor::new(s);

        // just checks if unwrap doesn't panic
        CsvReader::new(file)
            // we also check if infer schema ignores errors
            .infer_schema(Some(10))
            .has_header(true)
            .with_ignore_parser_errors(true)
            .finish()
            .unwrap();

        let s = r#""sepal.length","sepal.width","petal.length","petal.width","variety"
        5.1,3.5,1.4,.2,"Setosa"
        4.9,3,1.4,.2,"Setosa"
        4.7,3.2,1.3,.2,"Setosa"
        4.6,3.1,1.5,.2,"Setosa"
        5,3.6,1.4,.2,"Setosa"
        5.4,3.9,1.7,.4,"Setosa"
        4.6,3.4,1.4,.3,"Setosa"
"#;

        let file = Cursor::new(s);
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .finish()
            .unwrap();

        let col = df.column("variety").unwrap();
        dbg!(&df);
        assert_eq!(col.get(0), AnyValue::Utf8("Setosa"));
        assert_eq!(col.get(2), AnyValue::Utf8("Setosa"));

        assert_eq!("sepal.length", df.get_columns()[0].name());
        assert_eq!(1, df.column("sepal.length").unwrap().chunks().len());
        assert_eq!(df.height(), 7);

        // test windows line endings
        let s = "head_1,head_2\r\n1,2\r\n1,2\r\n1,2\r\n";

        let file = Cursor::new(s);
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .finish()
            .unwrap();

        assert_eq!("head_1", df.get_columns()[0].name());
        assert_eq!(df.shape(), (3, 2));
    }

    #[test]
    fn test_tab_sep() {
        let csv = br#"1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99217	Hospital observation care on day of discharge	N	68	67	68	73.821029412	381.30882353	57.880294118	58.2125
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99218	Hospital observation care, typically 30 minutes	N	19	19	19	100.88315789	476.94736842	76.795263158	77.469473684
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99220	Hospital observation care, typically 70 minutes	N	26	26	26	188.11076923	1086.9230769	147.47923077	147.79346154
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99221	Initial hospital inpatient care, typically 30 minutes per day	N	24	24	24	102.24	474.58333333	80.155	80.943333333
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99222	Initial hospital inpatient care, typically 50 minutes per day	N	17	17	17	138.04588235	625	108.22529412	109.22
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99223	Initial hospital inpatient care, typically 70 minutes per day	N	86	82	86	204.85395349	1093.5	159.25906977	161.78093023
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99232	Subsequent hospital inpatient care, typically 25 minutes per day	N	360	206	360	73.565666667	360.57222222	57.670305556	58.038833333
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99233	Subsequent hospital inpatient care, typically 35 minutes per day	N	284	148	284	105.34971831	576.98943662	82.512992958	82.805774648
"#.as_ref();

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .with_delimiter(b'\t')
            .has_header(false)
            .with_ignore_parser_errors(true)
            .finish()
            .unwrap();

        dbg!(df);
    }

    #[test]
    fn test_projection() {
        let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
        let df = CsvReader::from_path(path)
            .unwrap()
            .with_projection(Some(vec![0, 2]))
            .finish()
            .unwrap();
        dbg!(&df);
        let col_1 = df.select_at_idx(0).unwrap();
        assert_eq!(col_1.get(0), AnyValue::Utf8("vegetables"));
        assert_eq!(col_1.get(1), AnyValue::Utf8("seafood"));
        assert_eq!(col_1.get(2), AnyValue::Utf8("meat"));

        let col_2 = df.select_at_idx(1).unwrap();
        assert_eq!(col_2.get(0), AnyValue::Float64(0.5));
        assert_eq!(col_2.get(1), AnyValue::Float64(5.0));
        assert_eq!(col_2.get(2), AnyValue::Float64(5.0));
    }

    #[test]
    fn test_missing_data() {
        // missing data should not lead to parser error.
        let csv = r#"column_1,column_2,column_3
        1,2,3
        1,,3
"#;

        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish().unwrap();
        assert!(df
            .column("column_1")
            .unwrap()
            .series_equal(&Series::new("column_1", &[1, 1])));
        assert!(df
            .column("column_2")
            .unwrap()
            .series_equal_missing(&Series::new("column_2", &[Some(2), None])));
        assert!(df
            .column("column_3")
            .unwrap()
            .series_equal(&Series::new("column_3", &[3, 3])));
    }

    #[test]
    fn test_escape_comma() {
        let csv = r#"column_1,column_2,column_3
-86.64408227,"Autauga, Alabama, US",11
-86.64408227,"Autauga, Alabama, US",12
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish().unwrap();
        assert_eq!(df.shape(), (2, 3));
        assert!(df
            .column("column_3")
            .unwrap()
            .series_equal(&Series::new("column_3", &[11, 12])));
    }

    #[test]
    fn test_escape_double_quotes() {
        let csv = r#"column_1,column_2,column_3
-86.64408227,"with ""double quotes"" US",11
-86.64408227,"with ""double quotes followed"", by comma",12
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish().unwrap();
        println!("GET: {}", df.column("column_2").unwrap().get(1));
        println!("EXPECTEd: {}", r#"with "double quotes followed", by comma"#);
        assert_eq!(df.shape(), (2, 3));
        assert!(df.column("column_2").unwrap().series_equal(&Series::new(
            "column_2",
            &[
                r#"with "double quotes" US"#,
                r#"with "double quotes followed", by comma"#
            ]
        )));
    }

    #[test]
    fn test_escape_2() {
        // this is is harder than it looks.
        // Fields:
        // * hello
        // * ","
        // * " "
        // * world
        // * "!"
        let csv = r#"hello,","," ",world,"!"
hello,","," ",world,"!"
hello,","," ",world,"!"
hello,","," ",world,"!"
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(false)
            .with_n_threads(Some(1))
            .finish()
            .unwrap();

        for (col, val) in &[
            ("column_1", "hello"),
            ("column_2", ","),
            ("column_3", " "),
            ("column_4", "world"),
            ("column_5", "!"),
        ] {
            assert!(df
                .column(col)
                .unwrap()
                .series_equal(&Series::new(col, &[&**val; 4])));
        }
    }

    #[test]
    fn test_very_long_utf8() {
        let csv = r#"column_1,column_2,column_3
-86.64408227,"Lorem Ipsum is simply dummy text of the printing and typesetting
industry. Lorem Ipsum has been the industry's standard dummy text ever since th
e 1500s, when an unknown printer took a galley of type and scrambled it to make
a type specimen book. It has survived not only five centuries, but also the leap
into electronic typesetting, remaining essentially unchanged. It was popularised
in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including
versions of Lorem Ipsum.",11
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish().unwrap();

        assert!(df.column("column_2").unwrap().series_equal(&Series::new(
            "column_2",
            &[
                r#"Lorem Ipsum is simply dummy text of the printing and typesetting
industry. Lorem Ipsum has been the industry's standard dummy text ever since th
e 1500s, when an unknown printer took a galley of type and scrambled it to make
a type specimen book. It has survived not only five centuries, but also the leap
into electronic typesetting, remaining essentially unchanged. It was popularised
in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including
versions of Lorem Ipsum."#,
            ]
        )));
    }

    #[test]
    fn test_nulls_parser() {
        // test it does not fail on the leading comma.
        let csv = r#"id1,id2,id3,id4,id5,id6,v1,v2,v3
id047,id023,id0000084849,90,96,35790,2,9,93.348148
,id022,id0000031441,50,44,71525,3,11,81.013682
id090,id048,id0000067778,24,2,51862,4,9,
"#;

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(true)
            .with_n_threads(Some(1))
            .finish()
            .unwrap();
        assert_eq!(df.shape(), (3, 9));
    }

    #[test]
    fn test_new_line_escape() {
        let s = r#"
 "sepal.length","sepal.width","petal.length","petal.width","variety"
 5.1,3.5,1.4,.2,"Setosa
 texts after new line character"
 4.9,3,1.4,.2,"Setosa"
 "#;

        let file = Cursor::new(s);
        let _df = CsvReader::new(file).has_header(true).finish().unwrap();
    }

    #[test]
    fn test_quoted_numeric() {
        // CSV fields may be quoted
        let s = r#""foo","bar"
"4.9","3"
"1.4","2"
"#;

        let file = Cursor::new(s);
        let df = CsvReader::new(file).has_header(true).finish().unwrap();
        assert_eq!(df.column("bar").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("foo").unwrap().dtype(), &DataType::Float64);
    }

    #[test]
    fn test_empty_bytes_to_dataframe() {
        let fields = vec![Field::new("test_field", DataType::Utf8)];
        let schema = Schema::new(fields);
        let file = Cursor::new(vec![]);

        let result = CsvReader::new(file)
            .has_header(false)
            .with_columns(Some(
                schema
                    .fields()
                    .iter()
                    .map(|s| s.name().to_string())
                    .collect(),
            ))
            .with_schema(&schema)
            .finish();
        assert!(result.is_ok())
    }

    #[test]
    fn test_carriage_return() {
        let csv =
            "\"foo\",\"bar\"\r\n\"158252579.00\",\"7.5800\"\r\n\"158252579.00\",\"7.5800\"\r\n";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(true)
            .with_n_threads(Some(1))
            .finish()
            .unwrap();
        assert_eq!(df.shape(), (2, 2));
    }

    #[test]
    fn test_missing_value() {
        let csv = r#"foo,bar,ham
1,2,3
1,2,3
1,2
"#;

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(true)
            .with_schema(&Schema::new(vec![
                Field::new("foo", DataType::UInt32),
                Field::new("bar", DataType::UInt32),
                Field::new("ham", DataType::UInt32),
            ]))
            .finish()
            .unwrap();
        assert_eq!(df.column("ham").unwrap().len(), 3)
    }

    #[test]
    #[cfg(feature = "temporal")]
    fn test_with_dtype() -> Result<()> {
        // test if timestamps can be parsed as Datetime
        let csv = r#"a,b,c,d,e
AUDCAD,1616455919,0.91212,0.95556,1
AUDCAD,1616455920,0.92212,0.95556,1
AUDCAD,1616455921,0.96212,0.95666,1
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(true)
            .with_dtypes(Some(&Schema::new(vec![Field::new(
                "b",
                DataType::Datetime,
            )])))
            .finish()?;

        assert_eq!(
            df.dtypes(),
            &[
                DataType::Utf8,
                DataType::Datetime,
                DataType::Float64,
                DataType::Float64,
                DataType::Int64
            ]
        );
        Ok(())
    }

    #[test]
    fn test_skip_rows() -> Result<()> {
        let csv = r"#doc source pos typeindex type topic
#alpha : 25.0 25.0
#beta : 0.1
0 NA 0 0 57 0
0 NA 0 0 57 0
0 NA 5 5 513 0
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(false)
            .with_skip_rows(3)
            .with_delimiter(b' ')
            .finish()?;

        assert_eq!(df.height(), 3);
        Ok(())
    }

    #[test]
    fn test_projection_idx() -> Result<()> {
        let csv = r"#0 NA 0 0 57 0
0 NA 0 0 57 0
0 NA 5 5 513 0
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(false)
            .with_projection(Some(vec![4, 5]))
            .with_delimiter(b' ')
            .finish()?;

        assert_eq!(df.width(), 2);

        // this should give out of bounds error
        let file = Cursor::new(csv);
        let out = CsvReader::new(file)
            .has_header(false)
            .with_projection(Some(vec![4, 6]))
            .with_delimiter(b' ')
            .finish();

        assert!(out.is_err());
        Ok(())
    }

    #[test]
    fn test_missing_fields() -> Result<()> {
        let csv = r"1,2,3,4,5
1,2,3
1,2,3,4,5
1,3,5
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file).has_header(false).finish()?;

        use polars_core::df;
        let expect = df![
            "column_1" => [1, 1, 1, 1],
            "column_2" => [2, 2, 2, 3],
            "column_3" => [3, 3, 3, 5],
            "column_4" => [Some(4), None, Some(4), None],
            "column_5" => [Some(5), None, Some(5), None]
        ]?;
        assert!(df.frame_equal_missing(&expect));
        Ok(())
    }

    #[test]
    fn test_comment_lines() -> Result<()> {
        let csv = r"1,2,3,4,5
# this is a comment
1,2,3,4,5
# this is also a comment
1,2,3,4,5
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(false)
            .with_comment_char(Some(b'#'))
            .finish()?;
        assert_eq!(df.shape(), (3, 5));

        let csv = r"a,b,c,d,e
1,2,3,4,5
% this is a comment
1,2,3,4,5
% this is also a comment
1,2,3,4,5
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(true)
            .with_comment_char(Some(b'%'))
            .finish()?;
        assert_eq!(df.shape(), (3, 5));

        Ok(())
    }

    #[test]
    fn test_null_values_argument() -> Result<()> {
        let csv = r"1,a,foo
null-value,b,bar,
3,null-value,ham
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .has_header(false)
            .with_null_values(NullValues::AllColumns("null-value".to_string()).into())
            .finish()?;
        assert!(df.get_columns()[0].null_count() > 0);
        Ok(())
    }

    #[test]
    fn test_no_newline_at_end() -> Result<()> {
        let csv = r"a,b
foo,foo
bar,bar";
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish()?;

        use polars_core::df;
        let expect = df![
            "a" => ["foo", "bar"],
            "b" => ["foo", "bar"]
        ]?;
        assert!(df.frame_equal(&expect));
        Ok(())
    }

    #[test]
    #[cfg(feature = "temporal")]
    fn test_automatic_datetime_parsing() -> Result<()> {
        let csv = r"timestamp,open,high
2021-01-01 00:00:00,0.00305500,0.00306000
2021-01-01 00:15:00,0.00298800,0.00300400
2021-01-01 00:30:00,0.00298300,0.00300100
2021-01-01 00:45:00,0.00299400,0.00304000
";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file).with_parse_dates(true).finish()?;

        let ts = df.column("timestamp")?;
        assert_eq!(ts.dtype(), &DataType::Datetime);
        assert_eq!(ts.null_count(), 0);

        Ok(())
    }

    #[test]
    fn test_no_quotes() -> Result<()> {
        let rolling_stones = r#"
linenum,last_name,first_name
1,Jagger,Mick
2,O"Brian,Mary
3,Richards,Keith
4,L"Etoile,Bennet
5,Watts,Charlie
6,Smith,D"Shawn
7,Wyman,Bill
8,Woods,Ron
9,Jones,Brian
"#;

        let file = Cursor::new(rolling_stones);
        let df = CsvReader::new(file).with_quote_char(None).finish()?;
        assert_eq!(df.shape(), (9, 3));

        Ok(())
    }

    #[test]
    fn test_utf8() -> Result<()> {
        // first part is valid ascii. later we have removed some bytes from the emoji.
        let invalid_utf8 = [
            111, 10, 98, 97, 114, 10, 104, 97, 109, 10, 115, 112, 97, 109, 10, 106, 97, 109, 10,
            107, 97, 109, 10, 108, 97, 109, 10, 207, 128, 10, 112, 97, 109, 10, 115, 116, 97, 109,
            112, 10, 240, 159, 137, 10, 97, 115, 99, 105, 105, 10, 240, 159, 144, 172, 10, 99, 105,
            97, 111,
        ];
        let file = Cursor::new(invalid_utf8);
        assert!(CsvReader::new(file).finish().is_err());

        Ok(())
    }

    #[test]
    fn test_header_inference() -> Result<()> {
        let csv = r#"not_a_header,really,even_if,it_looks_like_one
1,2,3,4
4,3,2,1
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).has_header(false).finish()?;
        assert_eq!(df.dtypes(), vec![DataType::Utf8; 4]);
        Ok(())
    }

    #[test]
    fn test_header_with_comments() -> Result<()> {
        let csv = "# ignore me\na,b,c\nd,e,f";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .with_comment_char(Some(b'#'))
            .finish()?;
        // 1 row.
        assert_eq!(df.shape(), (1, 3));

        Ok(())
    }

    #[test]
    #[cfg(feature = "temporal")]
    fn test_ignore_parse_dates() -> Result<()> {
        // if parse dates is set, a given schema should still prevale above date parsing.
        let csv = r#"a,b,c
1,i,16200126
2,j,16250130
3,k,17220012
4,l,17290009"#;

        use DataType::*;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .with_parse_dates(true)
            .with_dtypes_slice(Some(&[Utf8, Utf8, Utf8]))
            .finish()?;

        assert_eq!(df.dtypes(), &[Utf8, Utf8, Utf8]);
        Ok(())
    }

    #[test]
    fn test_projection_and_quoting() -> Result<()> {
        let csv = "a,b,c,d
A1,'B1',C1,1
A2,\"B2\",C2,2
A3,\"B3\",C3,3
A3,\"B4_\"\"with_embedded_double_quotes\"\"\",C4,4";

        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish()?;
        assert_eq!(df.shape(), (4, 4));

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .with_n_threads(Some(1))
            .with_projection(Some(vec![0, 2]))
            .finish()?;
        assert_eq!(df.shape(), (4, 2));

        let file = Cursor::new(csv);
        let df = CsvReader::new(file)
            .with_n_threads(Some(1))
            .with_projection(Some(vec![1]))
            .finish()?;
        assert_eq!(df.shape(), (4, 1));

        Ok(())
    }

    #[test]
    fn test_infer_schema_0_rows() -> Result<()> {
        let csv = r#"a,b,c,d
1,a,1.0,true
1,a,1.0,false
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).infer_schema(Some(0)).finish()?;
        assert_eq!(
            df.dtypes(),
            &[
                DataType::Utf8,
                DataType::Utf8,
                DataType::Utf8,
                DataType::Utf8
            ]
        );
        Ok(())
    }

    #[test]
    fn test_infer_schema_eol() -> Result<()> {
        // no eol after header
        let no_eol = "colx,coly\nabcdef,1234";
        let file = Cursor::new(no_eol);
        let df = CsvReader::new(file).finish()?;
        assert_eq!(df.dtypes(), &[DataType::Utf8, DataType::Int64,]);
        Ok(())
    }

    #[test]
    fn test_whitespace_delimiters() -> Result<()> {
        let tsv = "\ta\tb\tc\n1\ta1\tb1\tc1\n2\ta2\tb2\tc2\n".to_string();
        let mut contents = Vec::with_capacity(3);
        contents.push((tsv.replace('\t', " "), b' '));
        contents.push((tsv.replace('\t', "-"), b'-'));
        contents.push((tsv, b'\t'));

        for (content, sep) in contents {
            let file = Cursor::new(&content);
            let df = CsvReader::new(file).with_delimiter(sep).finish()?;

            assert_eq!(df.shape(), (2, 4));
            assert_eq!(df.get_column_names(), &["", "a", "b", "c"]);
        }

        Ok(())
    }

    #[test]
    fn test_scientific_floats() -> Result<()> {
        let csv = r#"foo,bar
10000001,1e-5
10000002,.04
"#;
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).finish()?;
        assert_eq!(df.shape(), (2, 2));
        assert_eq!(df.dtypes(), &[DataType::Int64, DataType::Float64]);

        Ok(())
    }

    #[test]
    fn test_tsv_header_offset() -> Result<()> {
        let csv = "foo\tbar\n\t1000011\t1\n\t1000026\t2\n\t1000949\t2";
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).with_delimiter(b'\t').finish()?;

        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.dtypes(), &[DataType::Utf8, DataType::Int64]);
        let a = df.column("foo")?;
        let a = a.utf8()?;
        assert_eq!(a.get(0), Some(""));

        Ok(())
    }
}
