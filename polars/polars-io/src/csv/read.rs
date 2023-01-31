use super::*;
use crate::csv::read_impl::{to_batched_owned, BatchedCsvReader, OwnedBatchedCsvReader};
use crate::csv::utils::infer_file_schema;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CsvEncoding {
    /// Utf8 encoding
    Utf8,
    /// Utf8 encoding and unknown bytes are replaced with �
    LossyUtf8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NullValues {
    /// A single value that's used for all columns
    AllColumnsSingle(String),
    /// Multiple values that are used for all columns
    AllColumns(Vec<String>),
    /// Tuples that map column names to null value of that column
    Named(Vec<(String, String)>),
}

pub(super) enum NullValuesCompiled {
    /// A single value that's used for all columns
    AllColumnsSingle(String),
    // Multiple null values that are null for all columns
    AllColumns(Vec<String>),
    /// A different null value per column, computed from `NullValues::Named`
    Columns(Vec<String>),
}

impl NullValuesCompiled {
    pub(super) fn apply_projection(&mut self, projections: &[usize]) {
        if let Self::Columns(nv) = self {
            let nv = projections
                .iter()
                .map(|i| std::mem::take(&mut nv[*i]))
                .collect::<Vec<_>>();

            *self = NullValuesCompiled::Columns(nv);
        }
    }

    /// Safety
    /// The caller must ensure that `index` is in bounds
    pub(super) unsafe fn is_null(&self, field: &[u8], index: usize) -> bool {
        use NullValuesCompiled::*;
        match self {
            AllColumnsSingle(v) => v.as_bytes() == field,
            AllColumns(v) => v.iter().any(|v| v.as_bytes() == field),
            Columns(v) => {
                debug_assert!(index < v.len());
                v.get_unchecked(index).as_bytes() == field
            }
        }
    }
}

impl NullValues {
    pub(super) fn compile(self, schema: &Schema) -> PolarsResult<NullValuesCompiled> {
        Ok(match self {
            NullValues::AllColumnsSingle(v) => NullValuesCompiled::AllColumnsSingle(v),
            NullValues::AllColumns(v) => NullValuesCompiled::AllColumns(v),
            NullValues::Named(v) => {
                let mut null_values = vec!["".to_string(); schema.len()];
                for (name, null_value) in v {
                    let i = schema.try_index_of(&name)?;
                    null_values[i] = null_value;
                }
                NullValuesCompiled::Columns(null_values)
            }
        })
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
/// fn example() -> PolarsResult<DataFrame> {
///     CsvReader::from_path("iris.csv")?
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
    skip_rows_before_header: usize,
    /// Optional indexes of the columns to project
    projection: Option<Vec<usize>>,
    /// Optional column names to project/ select.
    columns: Option<Vec<String>>,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_errors: bool,
    pub(crate) schema: Option<&'a Schema>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<PathBuf>,
    schema_overwrite: Option<&'a Schema>,
    dtype_overwrite: Option<&'a [DataType]>,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    comment_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValues>,
    missing_is_null: bool,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    quote_char: Option<u8>,
    skip_rows_after_header: usize,
    parse_dates: bool,
    row_count: Option<RowCount>,
    // temporary schema needed for batch lifetimes
    owned_schema: Option<Box<Schema>>,
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
    pub fn with_ignore_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
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

    /// Skip the first `n` rows during parsing. The header will be parsed at `n` lines.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows_before_header = skip_rows;
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

    pub fn with_end_of_line_char(mut self, eol_char: u8) -> Self {
        self.eol_char = eol_char;
        self
    }

    /// Set values that will be interpreted as missing/ null. Note that any value you set as null value
    /// will not be escaped, so if quotation marks are part of the null value you should include them.
    pub fn with_null_values(mut self, null_values: Option<NullValues>) -> Self {
        self.null_values = null_values;
        self
    }

    /// Treat missing fields as null.
    pub fn with_missing_is_null(mut self, missing_is_null: bool) -> Self {
        self.missing_is_null = missing_is_null;
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
    pub fn with_parse_dates(mut self, toggle: bool) -> Self {
        self.parse_dates = toggle;
        self
    }

    #[cfg(feature = "private")]
    pub fn with_predicate(mut self, predicate: Option<Arc<dyn PhysicalIoExpr>>) -> Self {
        self.predicate = predicate;
        self
    }
}

impl<'a> CsvReader<'a, File> {
    /// This is the recommended way to create a csv reader as this allows for fastest parsing.
    pub fn from_path<P: Into<PathBuf>>(path: P) -> PolarsResult<Self> {
        let path = resolve_homedir(&path.into());
        let f = std::fs::File::open(&path)?;
        Ok(Self::new(f).with_path(Some(path)))
    }
}

impl<'a, R: MmapBytesReader + 'a> CsvReader<'a, R> {
    fn core_reader<'b>(
        &'b mut self,
        schema: Option<&'b Schema>,
        to_cast: Vec<Field>,
    ) -> PolarsResult<CoreReader<'b>>
    where
        'a: 'b,
    {
        let reader_bytes = get_reader_bytes(&mut self.reader)?;
        CoreReader::new(
            reader_bytes,
            self.n_rows,
            self.skip_rows_before_header,
            std::mem::take(&mut self.projection),
            self.max_records,
            self.delimiter,
            self.has_header,
            self.ignore_errors,
            self.schema,
            std::mem::take(&mut self.columns),
            self.encoding,
            self.n_threads,
            schema,
            self.dtype_overwrite,
            self.sample_size,
            self.chunk_size,
            self.low_memory,
            self.comment_char,
            self.quote_char,
            self.eol_char,
            std::mem::take(&mut self.null_values),
            self.missing_is_null,
            std::mem::take(&mut self.predicate),
            to_cast,
            self.skip_rows_after_header,
            std::mem::take(&mut self.row_count),
            self.parse_dates,
        )
    }

    fn prepare_schema_overwrite(&self, overwriting_schema: &Schema) -> (Schema, Vec<Field>, bool) {
        // This branch we check if there are dtypes we cannot parse.
        // We only support a few dtypes in the parser and later cast to the required dtype
        let mut to_cast = Vec::with_capacity(overwriting_schema.len());

        let mut _has_categorical = false;

        #[allow(clippy::unnecessary_filter_map)]
        let fields = overwriting_schema.iter_fields().filter_map(|mut fld| {
            use DataType::*;
            match fld.data_type() {
                Time => {
                    to_cast.push(fld);
                    // let inference decide the column type
                    None
                }
                Int8 | Int16 | UInt8 | UInt16 => {
                    // We have not compiled these buffers, so we cast them later.
                    to_cast.push(fld.clone());
                    fld.coerce(DataType::Int32);
                    Some(fld)
                }
                #[cfg(feature = "dtype-categorical")]
                Categorical(_) => {
                    _has_categorical = true;
                    Some(fld)
                }
                _ => Some(fld),
            }
        });
        let schema = Schema::from(fields);
        (schema, to_cast, _has_categorical)
    }

    pub fn batched_borrowed(&'a mut self) -> PolarsResult<BatchedCsvReader<'a>> {
        if let Some(schema) = self.schema_overwrite {
            let (schema, to_cast, has_cat) = self.prepare_schema_overwrite(schema);
            self.owned_schema = Some(Box::new(schema));

            // safety
            // we boxed the schema and we refer to the boxed pointer
            // the schema will drop once self drops
            // so it is bound to 'a
            let schema = unsafe {
                std::mem::transmute::<Option<&Schema>, Option<&Schema>>(
                    self.owned_schema.as_ref().map(|b| b.as_ref()),
                )
            };

            let csv_reader = self.core_reader(schema, to_cast)?;
            csv_reader.batched(has_cat)
        } else {
            let csv_reader = self.core_reader(self.schema, vec![])?;
            csv_reader.batched(false)
        }
    }
}

impl<'a> CsvReader<'a, Box<dyn MmapBytesReader>> {
    pub fn batched(mut self, schema: Option<SchemaRef>) -> PolarsResult<OwnedBatchedCsvReader> {
        match schema {
            Some(schema) => Ok(to_batched_owned(self, schema)),
            None => {
                let reader_bytes = get_reader_bytes(&mut self.reader)?;

                let (inferred_schema, _, _) = infer_file_schema(
                    &reader_bytes,
                    self.delimiter.unwrap_or(b','),
                    self.max_records,
                    self.has_header,
                    None,
                    &mut self.skip_rows_before_header,
                    self.skip_rows_after_header,
                    self.comment_char,
                    self.quote_char,
                    self.eol_char,
                    self.null_values.as_ref(),
                    self.parse_dates,
                )?;
                let schema = Arc::new(inferred_schema);
                Ok(to_batched_owned(self, schema))
            }
        }
    }
}

impl<'a, R> SerReader<R> for CsvReader<'a, R>
where
    R: MmapBytesReader + 'a,
{
    /// Create a new CsvReader from a file/ stream
    fn new(reader: R) -> Self {
        CsvReader {
            reader,
            rechunk: true,
            n_rows: None,
            max_records: Some(128),
            skip_rows_before_header: 0,
            projection: None,
            delimiter: None,
            has_header: true,
            ignore_errors: false,
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
            eol_char: b'\n',
            null_values: None,
            missing_is_null: true,
            predicate: None,
            quote_char: Some(b'"'),
            skip_rows_after_header: 0,
            parse_dates: false,
            row_count: None,
            owned_schema: None,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> PolarsResult<DataFrame> {
        let rechunk = self.rechunk;
        let schema_overwrite = self.schema_overwrite;
        let dtype_overwrite = self.dtype_overwrite;
        let should_parse_dates = self.parse_dates;
        let low_memory = self.low_memory;

        #[cfg(feature = "dtype-categorical")]
        let mut _cat_lock = None;

        let mut df = if let Some(schema) = schema_overwrite {
            let (schema, to_cast, _has_cat) = self.prepare_schema_overwrite(schema);

            #[cfg(feature = "dtype-categorical")]
            if _has_cat {
                _cat_lock = Some(polars_core::IUseStringCache::new())
            }

            let mut csv_reader = self.core_reader(Some(&schema), to_cast)?;
            csv_reader.as_df()?
        } else {
            #[cfg(feature = "dtype-categorical")]
            {
                let has_cat = self
                    .schema
                    .map(|schema| {
                        schema
                            .iter_dtypes()
                            .any(|dtype| matches!(dtype, DataType::Categorical(_)))
                    })
                    .unwrap_or(false);
                if has_cat {
                    _cat_lock = Some(polars_core::IUseStringCache::new())
                }
            }
            let mut csv_reader = self.core_reader(self.schema, vec![])?;
            csv_reader.as_df()?
        };

        // Important that this rechunk is never done in parallel.
        // As that leads to great memory overhead.
        if rechunk && df.n_chunks() > 1 {
            if low_memory {
                df.as_single_chunk();
            } else {
                df.as_single_chunk_par();
            }
        }

        #[cfg(feature = "temporal")]
        // only needed until we also can parse time columns in place
        if should_parse_dates {
            // determine the schema that's given by the user. That should not be changed
            let fixed_schema = match (schema_overwrite, dtype_overwrite) {
                (Some(schema), _) => Cow::Borrowed(schema),
                (None, Some(dtypes)) => {
                    let fields = dtypes
                        .iter()
                        .zip(df.get_column_names())
                        .map(|(dtype, name)| Field::new(name, dtype.clone()));

                    Cow::Owned(Schema::from(fields))
                }
                _ => Cow::Owned(Schema::default()),
            };
            df = parse_dates(df, &fixed_schema)
        }
        Ok(df)
    }
}

#[cfg(feature = "temporal")]
fn parse_dates(mut df: DataFrame, fixed_schema: &Schema) -> DataFrame {
    let cols = std::mem::take(df.get_columns_mut())
        .into_par_iter()
        .map(|s| {
            if let Ok(ca) = s.utf8() {
                // don't change columns that are in the fixed schema.
                if fixed_schema.index_of(s.name()).is_some() {
                    return s;
                }

                #[cfg(feature = "dtype-time")]
                if let Ok(ca) = ca.as_time(None, false) {
                    return ca.into_series();
                }
                s
            } else {
                s
            }
        })
        .collect::<Vec<_>>();

    DataFrame::new_no_checks(cols)
}
