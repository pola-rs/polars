mod batched;

use std::borrow::Cow;
use std::fmt;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use batched::*;
use polars_arrow::array::*;
use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::POOL;
#[cfg(feature = "polars-time")]
use polars_time::prelude::*;
use polars_utils::flatten;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::csv::buffer::*;
use crate::csv::parser::*;
use crate::csv::read::NullValuesCompiled;
use crate::csv::utils::*;
use crate::csv::{CsvEncoding, NullValues};
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
use crate::utils::update_row_counts;
use crate::RowCount;

pub(crate) fn cast_columns(
    df: &mut DataFrame,
    to_cast: &[Field],
    parallel: bool,
) -> PolarsResult<()> {
    use DataType::*;

    let cast_fn = |s: &Series, fld: &Field| match (s.dtype(), fld.data_type()) {
        #[cfg(feature = "temporal")]
        (Utf8, Date) => s
            .utf8()
            .unwrap()
            .as_date(None, false)
            .map(|ca| ca.into_series()),
        #[cfg(feature = "temporal")]
        (Utf8, Datetime(tu, _)) => s
            .utf8()
            .unwrap()
            .as_datetime(None, *tu, false, false, false, None)
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
            .collect::<PolarsResult<Vec<_>>>()?;
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
    ignore_errors: bool,
    skip_rows_before_header: usize,
    // after the header, we need to take embedded lines into account
    skip_rows_after_header: usize,
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
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    to_cast: Vec<Field>,
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
        ignore_errors: bool,
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
        eol_char: u8,
        null_values: Option<NullValues>,
        missing_is_null: bool,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        to_cast: Vec<Field>,
        skip_rows_after_header: usize,
        row_count: Option<RowCount>,
        parse_dates: bool,
    ) -> PolarsResult<CoreReader<'a>> {
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
                {
                    // We keep track of the inferred schema bool
                    // In case the file is compressed this schema inference is wrong and has to be done
                    // again after decompression.
                    #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
                    if let Some(b) =
                        decompress(&reader_bytes, n_rows, delimiter, quote_char, eol_char)
                    {
                        reader_bytes = ReaderBytes::Owned(b);
                    }

                    let (inferred_schema, _, _) = infer_file_schema(
                        &reader_bytes,
                        delimiter,
                        max_records,
                        has_header,
                        schema_overwrite,
                        &mut skip_rows,
                        skip_rows_after_header,
                        comment_char,
                        quote_char,
                        eol_char,
                        null_values.as_ref(),
                        parse_dates,
                    )?;
                    Cow::Owned(inferred_schema)
                }
            }
        };
        if let Some(dtypes) = dtype_overwrite {
            let mut s = schema.into_owned();
            for (index, dt) in dtypes.iter().enumerate() {
                s.coerce_by_index(index, dt.clone()).unwrap();
            }
            schema = Cow::Owned(s);
        }

        // create a null value for every column
        let mut null_values = null_values.map(|nv| nv.compile(&schema)).transpose()?;

        if let Some(cols) = columns {
            let mut prj = Vec::with_capacity(cols.len());
            for col in cols {
                let i = schema.try_index_of(&col)?;
                prj.push(i);
            }

            // update null values with projection
            if let Some(nv) = null_values.as_mut() {
                nv.apply_projection(&prj);
            }

            projection = Some(prj);
        }

        Ok(CoreReader {
            reader_bytes: Some(reader_bytes),
            schema,
            projection,
            line_number: usize::from(has_header),
            ignore_errors,
            skip_rows_before_header: skip_rows,
            skip_rows_after_header,
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
            eol_char,
            null_values,
            missing_is_null,
            predicate,
            to_cast,
            row_count,
        })
    }

    fn find_starting_point<'b>(
        &self,
        mut bytes: &'b [u8],
        eol_char: u8,
    ) -> PolarsResult<(&'b [u8], Option<usize>)> {
        let starting_point_offset = bytes.as_ptr() as usize;

        // Skip all leading white space and the occasional utf8-bom
        bytes = skip_whitespace_exclude(skip_bom(bytes), self.delimiter);
        // \n\n can be a empty string row of a single column
        // in other cases we skip it.
        if self.schema.len() > 1 {
            bytes = skip_line_ending(bytes, eol_char)
        }

        // If there is a header we skip it.
        if self.has_header {
            bytes = skip_header(bytes, eol_char).0;
        }

        if self.skip_rows_before_header > 0 {
            for _ in 0..self.skip_rows_before_header {
                let pos = next_line_position_naive(bytes, eol_char)
                    .ok_or_else(|| PolarsError::NoData("not enough lines to skip".into()))?;
                bytes = &bytes[pos..];
            }
        }

        if self.skip_rows_after_header > 0 {
            for _ in 0..self.skip_rows_after_header {
                let pos = match bytes.first() {
                    Some(first) if Some(*first) == self.comment_char => {
                        next_line_position_naive(bytes, eol_char)
                    }
                    // we don't pass expected fields
                    // as we want to skip all rows
                    // no matter the no. of fields
                    _ => next_line_position(bytes, None, self.delimiter, self.quote_char, eol_char),
                }
                .ok_or_else(|| PolarsError::NoData("not enough lines to skip".into()))?;

                bytes = &bytes[pos..];
            }
        }

        let starting_point_offset = if bytes.is_empty() {
            None
        } else {
            Some(bytes.as_ptr() as usize - starting_point_offset)
        };

        Ok((bytes, starting_point_offset))
    }

    #[allow(clippy::type_complexity)]
    fn determine_file_chunks_and_statistics(
        &self,
        n_threads: &mut usize,
        bytes: &'a [u8],
        logging: bool,
        streaming: bool,
    ) -> PolarsResult<(Vec<(usize, usize)>, usize, usize, Option<usize>, &'a [u8])> {
        // Make the variable mutable so that we can reassign the sliced file to this variable.
        let (mut bytes, starting_point_offset) = self.find_starting_point(bytes, self.eol_char)?;

        // initial row guess. We use the line statistic to guess the number of rows to allocate
        let mut total_rows = 128;

        // if None, there are less then 128 rows in the file and the statistics don't matter that much
        if let Some((mean, std)) = get_line_stats(
            bytes,
            self.sample_size,
            self.eol_char,
            self.schema.len(),
            self.delimiter,
            self.quote_char,
        ) {
            if logging {
                eprintln!("avg line length: {mean}\nstd. dev. line length: {std}");
            }

            // x % upper bound of byte length per line assuming normally distributed
            let line_length_upper_bound = mean + 1.1 * std;
            total_rows = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;

            // if we only need to parse n_rows,
            // we first try to use the line statistics to estimate the total bytes we need to process
            if let Some(n_rows) = self.n_rows {
                total_rows = std::cmp::min(n_rows, total_rows);

                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_rows as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position(
                        &bytes[n_bytes..],
                        Some(self.schema.len()),
                        self.delimiter,
                        self.quote_char,
                        self.eol_char,
                    ) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
            if logging {
                eprintln!("initial row estimate: {total_rows}")
            }
        }
        if total_rows == 128 {
            *n_threads = 1;

            if logging {
                eprintln!("file < 128 rows, no statistics determined")
            }
        }

        let chunk_size = std::cmp::min(self.chunk_size, total_rows);
        let n_chunks = total_rows / chunk_size;
        if logging {
            eprintln!(
                "no. of chunks: {n_chunks} processed by: {n_threads} threads at 1 chunk/thread",
            );
        }

        let n_file_chunks = if streaming { n_chunks } else { *n_threads };

        // split the file by the nearest new line characters such that every thread processes
        // approximately the same number of rows.
        Ok((
            get_file_chunks(
                bytes,
                n_file_chunks,
                self.schema.len(),
                self.delimiter,
                self.quote_char,
                self.eol_char,
            ),
            chunk_size,
            total_rows,
            starting_point_offset,
            bytes,
        ))
    }

    fn get_projection(&mut self) -> Vec<usize> {
        // we also need to sort the projection to have predictable output.
        // the `parse_lines` function expects this.
        self.projection
            .take()
            .map(|mut v| {
                v.sort_unstable();
                v
            })
            .unwrap_or_else(|| (0..self.schema.len()).collect())
    }

    fn get_string_columns(&self, projection: &[usize]) -> PolarsResult<Vec<&str>> {
        // keep track of the maximum capacity that needs to be allocated for the utf8-builder
        // Per string column we keep a statistic of the maximum length of string bytes per chunk
        // We must the names, not the indexes, (the indexes are incorrect due to projection
        // pushdown)
        let mut str_columns = Vec::with_capacity(projection.len());
        for i in projection {
            let (name, dtype) = self.schema.get_index(*i).ok_or_else(||
                PolarsError::ComputeError(
                    format!("the given projection index: {} is out of bounds for csv schema with {} columns", i, self.schema.len()).into())
            )?;

            if dtype == &DataType::Utf8 {
                str_columns.push(name.as_str())
            }
        }
        Ok(str_columns)
    }

    fn init_string_size_stats(&self, str_columns: &[&str], capacity: usize) -> Vec<RunningSize> {
        // assume 10 chars per str
        // this is not updated in low memory mode
        let init_str_bytes = capacity * 10;
        str_columns
            .iter()
            .map(|_| RunningSize::new(init_str_bytes))
            .collect()
    }

    fn parse_csv(
        &mut self,
        mut n_threads: usize,
        bytes: &[u8],
        predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<DataFrame> {
        let logging = verbose();
        let (file_chunks, chunk_size, total_rows, starting_point_offset, bytes) =
            self.determine_file_chunks_and_statistics(&mut n_threads, bytes, logging, false)?;
        let projection = self.get_projection();
        let str_columns = self.get_string_columns(&projection)?;

        // If the number of threads given by the user is lower than our global thread pool we create
        // new one.
        #[cfg(not(target_family = "wasm"))]
        let owned_pool;
        #[cfg(not(target_family = "wasm"))]
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
        #[cfg(target_family = "wasm")] // use a pre-created pool for wasm
        let pool = &POOL;
        // An empty file with a schema should return an empty DataFrame with that schema
        if bytes.is_empty() {
            // TODO! add DataFrame::new_from_schema
            let buffers = init_buffers(
                &projection,
                0,
                &self.schema,
                &self.init_string_size_stats(&str_columns, 0),
                self.quote_char,
                self.encoding,
                self.ignore_errors,
            )?;
            let df = DataFrame::new_no_checks(
                buffers
                    .into_iter()
                    .map(|buf| buf.into_series())
                    .collect::<PolarsResult<_>>()?,
            );
            return Ok(df);
        }

        // all the buffers returned from the threads
        // Structure:
        //      the inner vec has got buffers from all the columns.
        if let Some(predicate) = predicate {
            let str_capacities = self.init_string_size_stats(&str_columns, chunk_size);
            let dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let delimiter = self.delimiter;
                        let schema = self.schema.as_ref();
                        let ignore_errors = self.ignore_errors;
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
                                schema,
                                &str_capacities,
                                self.quote_char,
                                self.encoding,
                                self.ignore_errors,
                            )?;

                            let local_bytes = &bytes[read..stop_at_nbytes];

                            last_read = read;
                            let offset = read + starting_point_offset.unwrap();
                            read += parse_lines(
                                local_bytes,
                                offset,
                                delimiter,
                                self.comment_char,
                                self.quote_char,
                                self.eol_char,
                                self.null_values.as_ref(),
                                self.missing_is_null,
                                projection,
                                &mut buffers,
                                ignore_errors,
                                chunk_size,
                                self.schema.len(),
                            )?;

                            let mut local_df = DataFrame::new_no_checks(
                                buffers
                                    .into_iter()
                                    .map(|buf| buf.into_series())
                                    .collect::<PolarsResult<_>>()?,
                            );
                            let current_row_count = local_df.height() as IdxSize;
                            if let Some(rc) = &self.row_count {
                                local_df.with_row_count_mut(&rc.name, Some(rc.offset));
                            };

                            cast_columns(&mut local_df, &self.to_cast, false)?;
                            let s = predicate.evaluate(&local_df)?;
                            let mask = s.bool()?;
                            local_df = local_df.filter(mask)?;

                            // update the running str bytes statistics
                            if !self.low_memory {
                                update_string_stats(&str_capacities, &str_columns, &local_df)?;
                            }
                            dfs.push((local_df, current_row_count));
                        }
                        Ok(dfs)
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;
            let mut dfs = flatten(&dfs, None);
            if self.row_count.is_some() {
                update_row_counts(&mut dfs, 0)
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

            let str_capacities = self.init_string_size_stats(&str_columns, capacity);

            let mut dfs = pool.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let mut df = read_chunk(
                            bytes,
                            self.delimiter,
                            self.schema.as_ref(),
                            self.ignore_errors,
                            &projection,
                            bytes_offset_thread,
                            self.quote_char,
                            self.eol_char,
                            self.comment_char,
                            chunk_size,
                            &str_capacities,
                            self.encoding,
                            self.null_values.as_ref(),
                            self.missing_is_null,
                            usize::MAX,
                            stop_at_nbytes,
                            starting_point_offset,
                        )?;

                        // update the running str bytes statistics
                        if !self.low_memory {
                            update_string_stats(&str_capacities, &str_columns, &df)?;
                        }

                        cast_columns(&mut df, &self.to_cast, false)?;
                        if let Some(rc) = &self.row_count {
                            df.with_row_count_mut(&rc.name, Some(rc.offset));
                        }
                        let n_read = df.height() as IdxSize;
                        Ok((df, n_read))
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;
            if self.row_count.is_some() {
                update_row_counts(&mut dfs, 0)
            }
            accumulate_dataframes_vertical(dfs.into_iter().map(|t| t.0))
        }
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(&mut self) -> PolarsResult<DataFrame> {
        let predicate = self.predicate.take();
        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        let reader_bytes = self.reader_bytes.take().unwrap();

        let mut df = self.parse_csv(n_threads, &reader_bytes, predicate.as_ref())?;

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

fn update_string_stats(
    str_capacities: &[RunningSize],
    str_columns: &[&str],
    local_df: &DataFrame,
) -> PolarsResult<()> {
    // update the running str bytes statistics
    for (str_index, name) in str_columns.iter().enumerate() {
        let ca = local_df.column(name)?.utf8()?;
        let str_bytes_len = ca.get_values_size();

        let _ = str_capacities[str_index].update(str_bytes_len);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_chunk(
    bytes: &[u8],
    delimiter: u8,
    schema: &Schema,
    ignore_errors: bool,
    projection: &[usize],
    bytes_offset_thread: usize,
    quote_char: Option<u8>,
    eol_char: u8,
    comment_char: Option<u8>,
    capacity: usize,
    str_capacities: &[RunningSize],
    encoding: CsvEncoding,
    null_values: Option<&NullValuesCompiled>,
    missing_is_null: bool,
    chunk_size: usize,
    stop_at_nbytes: usize,
    starting_point_offset: Option<usize>,
) -> PolarsResult<DataFrame> {
    let mut read = bytes_offset_thread;
    let mut buffers = init_buffers(
        projection,
        capacity,
        schema,
        str_capacities,
        quote_char,
        encoding,
        ignore_errors,
    )?;

    let mut last_read = usize::MAX;
    loop {
        if read >= stop_at_nbytes || read == last_read {
            break;
        }
        let local_bytes = &bytes[read..stop_at_nbytes];

        last_read = read;
        let offset = read + starting_point_offset.unwrap();
        read += parse_lines(
            local_bytes,
            offset,
            delimiter,
            comment_char,
            quote_char,
            eol_char,
            null_values,
            missing_is_null,
            projection,
            &mut buffers,
            ignore_errors,
            chunk_size,
            schema.len(),
        )?;
    }

    Ok(DataFrame::new_no_checks(
        buffers
            .into_iter()
            .map(|buf| buf.into_series())
            .collect::<PolarsResult<_>>()?,
    ))
}
