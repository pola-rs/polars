use crate::aggregations::ScanAggregation;
use crate::csv::{CsvEncoding, NullValues};
use crate::csv_core::utils::*;
use crate::csv_core::{buffer::*, parser::*};
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
use crate::utils::update_row_counts;
use crate::RowCount;
use polars_arrow::array::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::{prelude::*, POOL};
use polars_time::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::borrow::Cow;
use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicUsize, Arc};

pub(crate) fn cast_columns(df: &mut DataFrame, to_cast: &[Field], parallel: bool) -> Result<()> {
    use DataType::*;

    let cast_fn = |s: &Series, fld: &Field| match (s.dtype(), fld.data_type()) {
        #[cfg(feature = "temporal")]
        (Utf8, Date) => s.utf8().unwrap().as_date(None).map(|ca| ca.into_series()),
        #[cfg(feature = "temporal")]
        (Utf8, Datetime(tu, _)) => s
            .utf8()
            .unwrap()
            .as_datetime(None, *tu)
            .map(|ca| ca.into_series()),
        (_, dt) => s.cast(dt),
    };

    if parallel {
        let cols = df
            .get_columns()
            .iter()
            .map(|s| {
                if let Some(fld) = to_cast.iter().find(|fld| fld.name().as_str() == s.name()) {
                    cast_fn(s, fld)
                } else {
                    Ok(s.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        *df = DataFrame::new_no_checks(cols)
    } else {
        // cast to the original dtypes in the schema
        for fld in to_cast {
            df.try_apply(fld.name(), |s| cast_fn(s, fld))?;
        }
    }
    Ok(())
}

/// CSV file reader
pub(crate) struct CoreReader<'a> {
    reader_bytes: Option<ReaderBytes<'a>>,
    /// Explicit schema for the CSV file
    schema: Cow<'a, Schema>,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// Current line number, used in error reporting
    line_number: usize,
    ignore_parser_errors: bool,
    skip_rows: usize,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    has_header: bool,
    delimiter: u8,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    null_values: Option<Vec<String>>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&'a [ScanAggregation]>,
    to_cast: &'a [Field],
    row_count: Option<RowCount>,
}

impl<'a> fmt::Debug for CoreReader<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("line_number", &self.line_number)
            .finish()
    }
}

pub(crate) struct RunningSize {
    max: AtomicUsize,
    sum: AtomicUsize,
    count: AtomicUsize,
    last: AtomicUsize,
}

fn compute_size_hint(max: usize, sum: usize, count: usize, last: usize) -> usize {
    let avg = (sum as f32 / count as f32) as usize;
    let size = std::cmp::max(last, avg) as f32;
    if (max as f32) < (size * 1.5) {
        max
    } else {
        size as usize
    }
}
impl RunningSize {
    fn new(size: usize) -> Self {
        Self {
            max: AtomicUsize::new(size),
            sum: AtomicUsize::new(size),
            count: AtomicUsize::new(1),
            last: AtomicUsize::new(size),
        }
    }

    pub(crate) fn update(&self, size: usize) -> (usize, usize, usize, usize) {
        let max = self.max.fetch_max(size, Ordering::Release);
        let sum = self.sum.fetch_add(size, Ordering::Release);
        let count = self.count.fetch_add(1, Ordering::Release);
        let last = self.last.fetch_add(size, Ordering::Release);
        (
            max,
            sum / count,
            last,
            compute_size_hint(max, sum, count, last),
        )
    }

    pub(crate) fn size_hint(&self) -> usize {
        let max = self.max.load(Ordering::Acquire);
        let sum = self.sum.load(Ordering::Acquire);
        let count = self.count.load(Ordering::Acquire);
        let last = self.last.load(Ordering::Acquire);
        compute_size_hint(max, sum, count, last)
    }
}

impl<'a> CoreReader<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_rows: Option<usize>,
        mut skip_rows: usize,
        mut projection: Option<Vec<usize>>,
        max_records: Option<usize>,
        delimiter: Option<u8>,
        has_header: bool,
        ignore_parser_errors: bool,
        schema: Option<&'a Schema>,
        columns: Option<Vec<String>>,
        encoding: CsvEncoding,
        n_threads: Option<usize>,
        schema_overwrite: Option<&'a Schema>,
        dtype_overwrite: Option<&'a [DataType]>,
        sample_size: usize,
        chunk_size: usize,
        low_memory: bool,
        comment_char: Option<u8>,
        quote_char: Option<u8>,
        null_values: Option<NullValues>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&'a [ScanAggregation]>,
        to_cast: &'a [Field],
        skip_rows_after_header: usize,
        row_count: Option<RowCount>,
        parse_dates: bool,
    ) -> Result<CoreReader<'a>> {
        #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
        let mut reader_bytes = reader_bytes;

        #[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
        if is_compressed(&reader_bytes) {
            return Err(PolarsError::ComputeError("cannot read compressed csv file; compile with feature 'decompress' or 'decompress-fast'".into()));
        }

        // check if schema should be inferred
        let delimiter = delimiter.unwrap_or(b',');

        let mut schema = match schema {
            Some(schema) => Cow::Borrowed(schema),
            None => {
                #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
                {
                    // We keep track of the inferred schema bool
                    // In case the file is compressed this schema inference is wrong and has to be done
                    // again after decompression.
                    if let Some(b) = decompress(&reader_bytes) {
                        reader_bytes = ReaderBytes::Owned(b);
                    }

                    let (inferred_schema, _) = infer_file_schema(
                        &reader_bytes,
                        delimiter,
                        max_records,
                        has_header,
                        schema_overwrite,
                        &mut skip_rows,
                        comment_char,
                        quote_char,
                        null_values.as_ref(),
                        parse_dates,
                    )?;
                    Cow::Owned(inferred_schema)
                }
                #[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
                {
                    let (inferred_schema, _) = infer_file_schema(
                        &reader_bytes,
                        delimiter,
                        max_records,
                        has_header,
                        schema_overwrite,
                        &mut skip_rows,
                        comment_char,
                        quote_char,
                        null_values.as_ref(),
                        parse_dates,
                    )?;
                    Cow::Owned(inferred_schema)
                }
            }
        };
        skip_rows += skip_rows_after_header;
        if let Some(dtypes) = dtype_overwrite {
            let mut s = schema.into_owned();
            for (index, dt) in dtypes.iter().enumerate() {
                s.coerce_by_index(index, dt.clone()).unwrap();
            }
            schema = Cow::Owned(s);
        }

        // create a null value for every column
        let mut null_values = null_values.map(|nv| nv.process(&schema)).transpose()?;

        if let Some(cols) = columns {
            let mut prj = Vec::with_capacity(cols.len());
            for col in cols {
                let i = schema.try_index_of(&col)?;
                prj.push(i);
            }

            // update null values with projection
            null_values = null_values.map(|mut nv| {
                prj.iter()
                    .map(|i| std::mem::take(&mut nv[*i]))
                    .collect::<Vec<_>>()
            });

            projection = Some(prj);
        }

        Ok(CoreReader {
            reader_bytes: Some(reader_bytes),
            schema,
            projection,
            line_number: if has_header { 1 } else { 0 },
            ignore_parser_errors,
            skip_rows,
            n_rows,
            encoding,
            n_threads,
            has_header,
            delimiter,
            sample_size,
            chunk_size,
            low_memory,
            comment_char,
            quote_char,
            null_values,
            predicate,
            aggregate,
            to_cast,
            row_count,
        })
    }

    fn find_starting_point<'b>(&self, mut bytes: &'b [u8]) -> Result<&'b [u8]> {
        // Skip all leading white space and the occasional utf8-bom
        bytes = skip_whitespace(skip_bom(bytes));
        // \n\n can be a empty string row of a single column
        // in other cases we skip it.
        if self.schema.len() > 1 {
            bytes = skip_line_ending(bytes)
        }

        // If there is a header we skip it.
        if self.has_header {
            bytes = skip_header(bytes).0;
        }

        if self.skip_rows > 0 {
            for _ in 0..self.skip_rows {
                // This does not check embedding of new line chars in string quotes.
                // TODO create a state machine/ or use that of csv crate to skip lines with proper
                // escaping
                let pos = next_line_position_naive(bytes)
                    .ok_or_else(|| PolarsError::NoData("not enough lines to skip".into()))?;
                bytes = &bytes[pos..];
            }
        }
        Ok(bytes)
    }

    fn parse_csv(
        &mut self,
        mut n_threads: usize,
        bytes: &[u8],
        predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    ) -> Result<DataFrame> {
        let logging = std::env::var("POLARS_VERBOSE").is_ok();
        // Make the variable mutable so that we can reassign the sliced file to this variable.
        let mut bytes = self.find_starting_point(bytes)?;

        // initial row guess. We use the line statistic to guess the number of rows to allocate
        let mut total_rows = 128;

        // if None, there are less then 128 rows in the file and the statistics don't matter that much
        if let Some((mean, std)) = get_line_stats(bytes, self.sample_size) {
            if logging {
                eprintln!("avg line length: {}\nstd. dev. line length: {}", mean, std);
            }

            // x % upper bound of byte length per line assuming normally distributed
            let line_length_upper_bound = mean + 1.1 * std;
            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;

            // if we only need to parse n_rows,
            // we first try to use the line statistics the total bytes we need to process
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);

                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position(
                        &bytes[n_bytes..],
                        self.schema.len(),
                        self.delimiter,
                        self.quote_char,
                    ) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
            if logging {
                eprintln!("initial row estimate: {}", total_rows)
            }
        }
        if total_rows == 128 {
            n_threads = 1;

            if logging {
                eprintln!("file < 128 rows, no statistics determined")
            }
        }

        // we also need to sort the projection to have predictable output.
        // the `parse_lines` function expects this.
        let projection = self
            .projection
            .take()
            .map(|mut v| {
                v.sort_unstable();
                v
            })
            .unwrap_or_else(|| (0..self.schema.len()).collect());

        let chunk_size = std::cmp::min(self.chunk_size, total_rows);
        let n_chunks = total_rows / chunk_size;
        if logging {
            eprintln!(
                "no. of chunks: {} processed by: {} threads at 1 chunk/thread",
                n_chunks, n_threads
            );
        }

        // keep track of the maximum capacity that needs to be allocated for the utf8-builder
        // Per string column we keep a statistic of the maximum length of string bytes per chunk
        // We must the names, not the indexes, (the indexes are incorrect due to projection
        // pushdown)
        let mut str_columns = Vec::with_capacity(projection.len());
        for i in &projection {
            let (name, dtype) = self.schema.get_index(*i).ok_or_else(||
                PolarsError::ComputeError(
                    format!("the given projection index: {} is out of bounds for csv schema with {} columns", i, self.schema.len()).into())
                )?;

            if dtype == &DataType::Utf8 {
                str_columns.push(name)
            }
        }

        // split the file by the nearest new line characters such that every thread processes
        // approximately the same number of rows.
        let file_chunks = get_file_chunks(
            bytes,
            n_threads,
            self.schema.len(),
            self.delimiter,
            self.quote_char,
        );

        // If the number of threads given by the user is lower than our global thread pool we create
        // new one.
        let owned_pool;
        let pool = if POOL.current_num_threads() != n_threads {
            owned_pool = Some(
                ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .unwrap(),
            );
            owned_pool.as_ref().unwrap()
        } else {
            &POOL
        };

        // all the buffers returned from the threads
        // Structure:
        //      the inner vec has got buffers from all the columns.
        if predicate.is_some() {
            // assume 10 chars per str
            // this is not updated in low memory mode
            let init_str_bytes = chunk_size * 10;
            let str_capacities: Vec<_> = str_columns
                .iter()
                .map(|_| RunningSize::new(init_str_bytes))
                .collect();

            // An empty file with a schema should return an empty DataFrame with that schema
            if bytes.is_empty() {
                let buffers = init_buffers(
                    &projection,
                    0,
                    &self.schema,
                    &str_capacities,
                    self.quote_char,
                    self.encoding,
                    self.ignore_parser_errors,
                )?;
                let df = DataFrame::new_no_checks(
                    buffers
                        .into_iter()
                        .map(|buf| buf.into_series())
                        .collect::<Result<_>>()?,
                );
                return Ok(df);
            }

            let dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let delimiter = self.delimiter;
                        let schema = self.schema.clone();
                        let ignore_parser_errors = self.ignore_parser_errors;
                        let projection = &projection;

                        let mut read = bytes_offset_thread;
                        let mut dfs = Vec::with_capacity(256);

                        let mut last_read = usize::MAX;
                        loop {
                            if read >= stop_at_nbytes || read == last_read {
                                break;
                            }

                            let mut buffers = init_buffers(
                                projection,
                                chunk_size,
                                &schema,
                                &str_capacities,
                                self.quote_char,
                                self.encoding,
                                self.ignore_parser_errors,
                            )?;

                            let local_bytes = &bytes[read..stop_at_nbytes];

                            last_read = read;
                            read += parse_lines(
                                local_bytes,
                                read,
                                delimiter,
                                self.comment_char,
                                self.quote_char,
                                self.null_values.as_ref(),
                                projection,
                                &mut buffers,
                                ignore_parser_errors,
                                chunk_size,
                                self.schema.len(),
                            )?;

                            let mut local_df = DataFrame::new_no_checks(
                                buffers
                                    .into_iter()
                                    .map(|buf| buf.into_series())
                                    .collect::<Result<_>>()?,
                            );
                            let current_row_count = local_df.height() as u32;
                            if let Some(rc) = &self.row_count {
                                local_df.with_row_count_mut(&rc.name, Some(rc.offset));
                            };

                            if let Some(predicate) = predicate {
                                let s = predicate.evaluate(&local_df)?;
                                let mask =
                                    s.bool().expect("filter predicates was not of type boolean");
                                local_df = local_df.filter(mask)?;
                            }

                            // update the running str bytes statistics
                            for (str_index, name) in str_columns.iter().enumerate() {
                                let ca = local_df.column(name)?.utf8()?;
                                let str_bytes_len = ca.get_values_size();

                                // don't update running statistics if we try to reduce string memory usage.
                                if self.low_memory {
                                    local_df.shrink_to_fit();
                                    let (max, avg, last, size_hint) =
                                        str_capacities[str_index].update(str_bytes_len);
                                    if logging {
                                        if size_hint < str_bytes_len {
                                            eprintln!(
                                                "probably needed to reallocate column: {}\
                                    \nprevious capacity was: {}\
                                    \nneeded capacity was: {}",
                                                name, size_hint, str_bytes_len
                                            );
                                        }
                                        eprintln!(
                                            "column {} statistics: \nmax: {}\navg: {}\nlast: {}",
                                            name, max, avg, last
                                        )
                                    }
                                }
                            }
                            cast_columns(&mut local_df, self.to_cast, false)?;
                            dfs.push((local_df, current_row_count));
                        }
                        Ok(dfs)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            let mut dfs = dfs.into_iter().flatten().collect::<Vec<_>>();
            if self.row_count.is_some() {
                update_row_counts(&mut dfs)
            }
            accumulate_dataframes_vertical(dfs.into_iter().map(|t| t.0))
        } else {
            // let exponential growth solve the needed size. This leads to less memory overhead
            // in the later rechunk. Because we have large chunks they are easier reused for the
            // large final contiguous memory needed at the end.
            let rows_per_thread = total_rows / n_threads;
            let max_proxy = bytes.len() / n_threads / 2;
            let capacity = if self.low_memory {
                chunk_size
            } else {
                std::cmp::min(rows_per_thread, max_proxy)
            };

            // assume 10 chars per str
            let init_str_bytes = capacity * 10;
            let str_capacities: Vec<_> = str_columns
                .iter()
                .map(|_| RunningSize::new(init_str_bytes))
                .collect();

            let mut dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let delimiter = self.delimiter;
                        let schema = self.schema.clone();
                        let ignore_parser_errors = self.ignore_parser_errors;
                        let projection = &projection;

                        let mut read = bytes_offset_thread;
                        let mut buffers = init_buffers(
                            projection,
                            capacity,
                            &schema,
                            &str_capacities,
                            self.quote_char,
                            self.encoding,
                            self.ignore_parser_errors,
                        )?;

                        let mut last_read = usize::MAX;
                        loop {
                            if read >= stop_at_nbytes || read == last_read {
                                break;
                            }
                            let local_bytes = &bytes[read..stop_at_nbytes];

                            last_read = read;
                            read += parse_lines(
                                local_bytes,
                                read,
                                delimiter,
                                self.comment_char,
                                self.quote_char,
                                self.null_values.as_ref(),
                                projection,
                                &mut buffers,
                                ignore_parser_errors,
                                // chunk size doesn't really matter anymore,
                                // less calls if we increase the size
                                usize::MAX,
                                self.schema.len(),
                            )?;
                        }

                        let mut df = DataFrame::new_no_checks(
                            buffers
                                .into_iter()
                                .map(|buf| buf.into_series())
                                .collect::<Result<_>>()?,
                        );

                        cast_columns(&mut df, self.to_cast, false)?;
                        if let Some(rc) = &self.row_count {
                            df.with_row_count_mut(&rc.name, Some(rc.offset));
                        }
                        let n_read = df.height() as u32;
                        Ok((df, n_read))
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            if self.row_count.is_some() {
                update_row_counts(&mut dfs)
            }
            accumulate_dataframes_vertical(dfs.into_iter().map(|t| t.0))
        }
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(&mut self) -> Result<DataFrame> {
        let predicate = self.predicate.take();
        let aggregate = self.aggregate.take();
        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        let reader_bytes = self.reader_bytes.take().unwrap();

        let mut df = self.parse_csv(n_threads, &reader_bytes, predicate.as_ref())?;

        if let Some(aggregate) = aggregate {
            let cols = aggregate
                .iter()
                .map(|scan_agg| scan_agg.finish(&df))
                .collect::<Result<Vec<_>>>()?;
            df = DataFrame::new_no_checks(cols)
        }

        // if multi-threaded the n_rows was probabilistically determined.
        // Let's slice to correct number of rows if possible.
        if let Some(n_rows) = self.n_rows {
            if n_rows < df.height() {
                df = df.slice(0, n_rows)
            }
        }
        Ok(df)
    }
}
