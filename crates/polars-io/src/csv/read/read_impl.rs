pub(super) mod batched;

use std::fmt;

use polars_core::config::verbose;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, handle_casting_failures};
use polars_core::POOL;
#[cfg(feature = "polars-time")]
use polars_time::prelude::*;
use polars_utils::flatten;
use rayon::prelude::*;

use super::buffer::init_buffers;
use super::options::{CommentPrefix, CsvEncoding, NullValues, NullValuesCompiled};
use super::parser::{
    get_line_stats, is_comment_line, next_line_position, next_line_position_naive, parse_lines,
    skip_bom, skip_line_ending, skip_this_line, skip_whitespace_exclude,
};
use super::schema_inference::{check_decimal_comma, infer_file_schema};
#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
use super::utils::decompress;
use super::utils::get_file_chunks;
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
#[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
use crate::utils::is_compressed;
use crate::utils::update_row_counts;
use crate::RowIndex;

pub(crate) fn cast_columns(
    df: &mut DataFrame,
    to_cast: &[Field],
    parallel: bool,
    ignore_errors: bool,
) -> PolarsResult<()> {
    let cast_fn = |s: &Series, fld: &Field| {
        let out = match (s.dtype(), fld.data_type()) {
            #[cfg(feature = "temporal")]
            (DataType::String, DataType::Date) => s
                .str()
                .unwrap()
                .as_date(None, false)
                .map(|ca| ca.into_series()),
            #[cfg(feature = "temporal")]
            (DataType::String, DataType::Datetime(tu, _)) => s
                .str()
                .unwrap()
                .as_datetime(
                    None,
                    *tu,
                    false,
                    false,
                    None,
                    &StringChunked::from_iter(std::iter::once("raise")),
                )
                .map(|ca| ca.into_series()),
            (_, dt) => s.cast(dt),
        }?;
        if !ignore_errors && s.null_count() != out.null_count() {
            handle_casting_failures(s, &out)?;
        }
        Ok(out)
    };

    if parallel {
        let cols = POOL.install(|| {
            df.get_columns()
                .into_par_iter()
                .map(|s| {
                    if let Some(fld) = to_cast.iter().find(|fld| fld.name().as_str() == s.name()) {
                        cast_fn(s, fld)
                    } else {
                        Ok(s.clone())
                    }
                })
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        *df = unsafe { DataFrame::new_no_checks(cols) }
    } else {
        // cast to the original dtypes in the schema
        for fld in to_cast {
            // field may not be projected
            if let Some(idx) = df.get_column_index(fld.name()) {
                df.try_apply_at_idx(idx, |s| cast_fn(s, fld))?;
            }
        }
    }
    Ok(())
}

/// CSV file reader
pub(crate) struct CoreReader<'a> {
    reader_bytes: Option<ReaderBytes<'a>>,
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// Current line number, used in error reporting
    current_line: usize,
    ignore_errors: bool,
    skip_rows_before_header: usize,
    // after the header, we need to take embedded lines into account
    skip_rows_after_header: usize,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    has_header: bool,
    separator: u8,
    sample_size: usize,
    chunk_size: usize,
    low_memory: bool,
    decimal_comma: bool,
    comment_prefix: Option<CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValuesCompiled>,
    missing_is_null: bool,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    to_cast: Vec<Field>,
    row_index: Option<RowIndex>,
    truncate_ragged_lines: bool,
}

impl<'a> fmt::Debug for CoreReader<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("current_line", &self.current_line)
            .finish()
    }
}

impl<'a> CoreReader<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_rows: Option<usize>,
        skip_rows: usize,
        mut projection: Option<Vec<usize>>,
        max_records: Option<usize>,
        separator: Option<u8>,
        has_header: bool,
        ignore_errors: bool,
        schema: Option<SchemaRef>,
        columns: Option<Arc<[String]>>,
        encoding: CsvEncoding,
        mut n_threads: Option<usize>,
        schema_overwrite: Option<SchemaRef>,
        dtype_overwrite: Option<Arc<Vec<DataType>>>,
        sample_size: usize,
        chunk_size: usize,
        low_memory: bool,
        comment_prefix: Option<CommentPrefix>,
        quote_char: Option<u8>,
        eol_char: u8,
        null_values: Option<NullValues>,
        missing_is_null: bool,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        to_cast: Vec<Field>,
        skip_rows_after_header: usize,
        row_index: Option<RowIndex>,
        try_parse_dates: bool,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
        decimal_comma: bool,
    ) -> PolarsResult<CoreReader<'a>> {
        let separator = separator.unwrap_or(b',');

        check_decimal_comma(decimal_comma, separator)?;
        #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
        let mut reader_bytes = reader_bytes;

        #[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
        if is_compressed(&reader_bytes) {
            polars_bail!(
                ComputeError: "cannot read compressed CSV file; \
                compile with feature 'decompress' or 'decompress-fast'"
            );
        }
        // We keep track of the inferred schema bool
        // In case the file is compressed this schema inference is wrong and has to be done
        // again after decompression.
        #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
        {
            let total_n_rows =
                n_rows.map(|n| skip_rows + (has_header as usize) + skip_rows_after_header + n);
            if let Some(b) =
                decompress(&reader_bytes, total_n_rows, separator, quote_char, eol_char)
            {
                reader_bytes = ReaderBytes::Owned(b);
            }
        }

        let mut schema = match schema {
            Some(schema) => schema,
            None => {
                let (inferred_schema, _, _) = infer_file_schema(
                    &reader_bytes,
                    separator,
                    max_records,
                    has_header,
                    schema_overwrite.as_deref(),
                    skip_rows,
                    skip_rows_after_header,
                    comment_prefix.as_ref(),
                    quote_char,
                    eol_char,
                    null_values.as_ref(),
                    try_parse_dates,
                    raise_if_empty,
                    &mut n_threads,
                    decimal_comma,
                )?;
                Arc::new(inferred_schema)
            },
        };
        if let Some(dtypes) = dtype_overwrite {
            let s = Arc::make_mut(&mut schema);
            for (index, dt) in dtypes.iter().enumerate() {
                s.set_dtype_at_index(index, dt.clone()).unwrap();
            }
        }

        // Create a null value for every column
        let null_values = null_values.map(|nv| nv.compile(&schema)).transpose()?;

        if let Some(cols) = columns {
            let mut prj = Vec::with_capacity(cols.len());
            for col in cols.as_ref() {
                let i = schema.try_index_of(col)?;
                prj.push(i);
            }
            projection = Some(prj);
        }

        Ok(CoreReader {
            reader_bytes: Some(reader_bytes),
            schema,
            projection,
            current_line: usize::from(has_header),
            ignore_errors,
            skip_rows_before_header: skip_rows,
            skip_rows_after_header,
            n_rows,
            encoding,
            n_threads,
            has_header,
            separator,
            sample_size,
            chunk_size,
            low_memory,
            comment_prefix,
            quote_char,
            eol_char,
            null_values,
            missing_is_null,
            predicate,
            to_cast,
            row_index,
            truncate_ragged_lines,
            decimal_comma,
        })
    }

    fn find_starting_point<'b>(
        &self,
        mut bytes: &'b [u8],
        quote_char: Option<u8>,
        eol_char: u8,
    ) -> PolarsResult<(&'b [u8], Option<usize>)> {
        let starting_point_offset = bytes.as_ptr() as usize;

        // Skip all leading white space and the occasional utf8-bom
        bytes = skip_whitespace_exclude(skip_bom(bytes), self.separator);
        // \n\n can be a empty string row of a single column
        // in other cases we skip it.
        if self.schema.len() > 1 {
            bytes = skip_line_ending(bytes, eol_char)
        }

        // skip 'n' leading rows
        if self.skip_rows_before_header > 0 {
            for _ in 0..self.skip_rows_before_header {
                let pos = next_line_position_naive(bytes, eol_char)
                    .ok_or_else(|| polars_err!(NoData: "not enough lines to skip"))?;
                bytes = &bytes[pos..];
            }
        }

        // skip lines that are comments
        while is_comment_line(bytes, self.comment_prefix.as_ref()) {
            bytes = skip_this_line(bytes, quote_char, eol_char);
        }

        // skip header row
        if self.has_header {
            bytes = skip_this_line(bytes, quote_char, eol_char);
        }
        // skip 'n' rows following the header
        if self.skip_rows_after_header > 0 {
            for _ in 0..self.skip_rows_after_header {
                let pos = if is_comment_line(bytes, self.comment_prefix.as_ref()) {
                    next_line_position_naive(bytes, eol_char)
                } else {
                    // we don't pass expected fields
                    // as we want to skip all rows
                    // no matter the no. of fields
                    next_line_position(bytes, None, self.separator, self.quote_char, eol_char)
                }
                .ok_or_else(|| polars_err!(NoData: "not enough lines to skip"))?;

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

    /// Estimates number of rows and optionally ensure we don't read more than `n_rows`
    /// by slicing `bytes` to the upper bound.
    fn estimate_rows_and_set_upper_bound<'b>(
        &self,
        mut bytes: &'b [u8],
        logging: bool,
        set_upper_bound: bool,
    ) -> (&'b [u8], usize, Option<&'b [u8]>) {
        // initial row guess. We use the line statistic to guess the number of rows to allocate
        let mut total_rows = 128;

        // if we set an upper bound on bytes, keep a reference to the bytes beyond the bound
        let mut remaining_bytes = None;

        // if None, there are less then 128 rows in the file and the statistics don't matter that much
        if let Some((mean, std)) = get_line_stats(
            bytes,
            self.sample_size,
            self.eol_char,
            Some(self.schema.len()),
            self.separator,
            self.quote_char,
        ) {
            if logging {
                eprintln!("avg line length: {mean}\nstd. dev. line length: {std}");
            }

            // x % upper bound of byte length per line assuming normally distributed
            // this upper bound assumption is not guaranteed to be accurate
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
                        self.separator,
                        self.quote_char,
                        self.eol_char,
                    ) {
                        if set_upper_bound {
                            (bytes, remaining_bytes) =
                                (&bytes[..n_bytes + pos], Some(&bytes[n_bytes + pos..]))
                        }
                    }
                }
            }
            if logging {
                eprintln!("initial row estimate: {total_rows}")
            }
        }
        (bytes, total_rows, remaining_bytes)
    }

    #[allow(clippy::type_complexity)]
    fn determine_file_chunks_and_statistics(
        &self,
        n_threads: &mut usize,
        bytes: &'a [u8],
        logging: bool,
    ) -> PolarsResult<(
        Vec<(usize, usize)>,
        usize,
        usize,
        Option<usize>,
        &'a [u8],
        Option<&'a [u8]>,
    )> {
        // Make the variable mutable so that we can reassign the sliced file to this variable.
        let (bytes, starting_point_offset) =
            self.find_starting_point(bytes, self.quote_char, self.eol_char)?;

        let (bytes, total_rows, remaining_bytes) =
            self.estimate_rows_and_set_upper_bound(bytes, logging, true);
        if total_rows == 128 {
            *n_threads = 1;

            if logging {
                eprintln!("file < 128 rows, no statistics determined")
            }
        }

        let chunk_size = std::cmp::min(self.chunk_size, total_rows);
        let n_file_chunks = *n_threads;

        // split the file by the nearest new line characters such that every thread processes
        // approximately the same number of rows.

        let chunks = get_file_chunks(
            bytes,
            n_file_chunks,
            Some(self.schema.len()),
            self.separator,
            self.quote_char,
            self.eol_char,
        );

        if logging {
            eprintln!(
                "no. of chunks: {} processed by: {n_threads} threads.",
                chunks.len()
            );
        }

        Ok((
            chunks,
            chunk_size,
            total_rows,
            starting_point_offset,
            bytes,
            remaining_bytes,
        ))
    }

    fn get_projection(&mut self) -> PolarsResult<Vec<usize>> {
        // we also need to sort the projection to have predictable output.
        // the `parse_lines` function expects this.
        self.projection
            .take()
            .map(|mut v| {
                v.sort_unstable();
                if let Some(idx) = v.last() {
                    polars_ensure!(*idx < self.schema.len(), OutOfBounds: "projection index: {} is out of bounds for csv schema with length: {}", idx, self.schema.len())
                }
                Ok(v)
            })
            .unwrap_or_else(|| Ok((0..self.schema.len()).collect()))
    }

    fn parse_csv(
        &mut self,
        mut n_threads: usize,
        bytes: &[u8],
        predicate: Option<&Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<DataFrame> {
        let logging = verbose();
        let (file_chunks, chunk_size, total_rows, starting_point_offset, bytes, remaining_bytes) =
            self.determine_file_chunks_and_statistics(&mut n_threads, bytes, logging)?;
        let projection = self.get_projection()?;

        // An empty file with a schema should return an empty DataFrame with that schema
        if bytes.is_empty() {
            let mut df = if projection.len() == self.schema.len() {
                DataFrame::empty_with_schema(self.schema.as_ref())
            } else {
                DataFrame::empty_with_schema(
                    &projection
                        .iter()
                        .map(|&i| self.schema.get_at_index(i).unwrap())
                        .map(|(name, dtype)| Field {
                            name: name.clone(),
                            dtype: dtype.clone(),
                        })
                        .collect::<Schema>(),
                )
            };
            if let Some(ref row_index) = self.row_index {
                df.insert_column(0, Series::new_empty(&row_index.name, &IDX_DTYPE))?;
            }
            return Ok(df);
        }

        // all the buffers returned from the threads
        // Structure:
        //      the inner vec has got buffers from all the columns.
        if let Some(predicate) = predicate {
            let dfs = POOL.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
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
                                self.quote_char,
                                self.encoding,
                                self.decimal_comma,
                            )?;

                            let local_bytes = &bytes[read..stop_at_nbytes];

                            last_read = read;
                            let offset = read + starting_point_offset.unwrap();
                            read += parse_lines(
                                local_bytes,
                                offset,
                                self.separator,
                                self.comment_prefix.as_ref(),
                                self.quote_char,
                                self.eol_char,
                                self.missing_is_null,
                                ignore_errors,
                                self.truncate_ragged_lines,
                                self.null_values.as_ref(),
                                projection,
                                &mut buffers,
                                chunk_size,
                                self.schema.len(),
                                &self.schema,
                            )?;

                            let columns = buffers
                                .into_iter()
                                .map(|buf| buf.into_series())
                                .collect::<PolarsResult<_>>()?;
                            let mut local_df = unsafe { DataFrame::new_no_checks(columns) };
                            let current_row_count = local_df.height() as IdxSize;
                            if let Some(rc) = &self.row_index {
                                local_df.with_row_index_mut(&rc.name, Some(rc.offset));
                            };

                            cast_columns(&mut local_df, &self.to_cast, false, self.ignore_errors)?;
                            let s = predicate.evaluate_io(&local_df)?;
                            let mask = s.bool()?;
                            local_df = local_df.filter(mask)?;

                            dfs.push((local_df, current_row_count));
                        }
                        Ok(dfs)
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;
            let mut dfs = flatten(&dfs, None);
            if self.row_index.is_some() {
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

            let mut dfs = POOL.install(|| {
                file_chunks
                    .into_par_iter()
                    .map(|(bytes_offset_thread, stop_at_nbytes)| {
                        let mut df = read_chunk(
                            bytes,
                            self.separator,
                            self.schema.as_ref(),
                            self.ignore_errors,
                            &projection,
                            bytes_offset_thread,
                            self.quote_char,
                            self.eol_char,
                            self.comment_prefix.as_ref(),
                            capacity,
                            self.encoding,
                            self.null_values.as_ref(),
                            self.missing_is_null,
                            self.truncate_ragged_lines,
                            usize::MAX,
                            stop_at_nbytes,
                            starting_point_offset,
                            self.decimal_comma,
                        )?;

                        cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;
                        if let Some(rc) = &self.row_index {
                            df.with_row_index_mut(&rc.name, Some(rc.offset));
                        }
                        let n_read = df.height() as IdxSize;
                        Ok((df, n_read))
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?;
            if let (Some(n_rows), Some(remaining_bytes)) = (self.n_rows, remaining_bytes) {
                let rows_already_read: usize = dfs.iter().map(|x| x.1 as usize).sum();
                if rows_already_read < n_rows {
                    dfs.push({
                        let mut df = {
                            let remaining_rows = n_rows - rows_already_read;
                            let mut buffers = init_buffers(
                                &projection,
                                remaining_rows,
                                self.schema.as_ref(),
                                self.quote_char,
                                self.encoding,
                                self.decimal_comma,
                            )?;

                            parse_lines(
                                remaining_bytes,
                                0,
                                self.separator,
                                self.comment_prefix.as_ref(),
                                self.quote_char,
                                self.eol_char,
                                self.missing_is_null,
                                self.ignore_errors,
                                self.truncate_ragged_lines,
                                self.null_values.as_ref(),
                                &projection,
                                &mut buffers,
                                remaining_rows - 1,
                                self.schema.len(),
                                self.schema.as_ref(),
                            )?;

                            let columns = buffers
                                .into_iter()
                                .map(|buf| buf.into_series())
                                .collect::<PolarsResult<_>>()?;
                            unsafe { DataFrame::new_no_checks(columns) }
                        };

                        cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;
                        if let Some(rc) = &self.row_index {
                            df.with_row_index_mut(&rc.name, Some(rc.offset));
                        }
                        let n_read = df.height() as IdxSize;
                        (df, n_read)
                    });
                }
            }
            if self.row_index.is_some() {
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

#[allow(clippy::too_many_arguments)]
fn read_chunk(
    bytes: &[u8],
    separator: u8,
    schema: &Schema,
    ignore_errors: bool,
    projection: &[usize],
    bytes_offset_thread: usize,
    quote_char: Option<u8>,
    eol_char: u8,
    comment_prefix: Option<&CommentPrefix>,
    capacity: usize,
    encoding: CsvEncoding,
    null_values: Option<&NullValuesCompiled>,
    missing_is_null: bool,
    truncate_ragged_lines: bool,
    chunk_size: usize,
    stop_at_nbytes: usize,
    starting_point_offset: Option<usize>,
    decimal_comma: bool,
) -> PolarsResult<DataFrame> {
    let mut read = bytes_offset_thread;
    // There's an off-by-one error somewhere in the reading code, where it reads
    // one more item than the requested capacity. Given the batch sizes are
    // approximate (sometimes they're smaller), this isn't broken, but it does
    // mean a bunch of extra allocation and copying. So we allocate a
    // larger-by-one buffer so the size is more likely to be accurate.
    let mut buffers = init_buffers(
        projection,
        capacity + 1,
        schema,
        quote_char,
        encoding,
        decimal_comma,
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
            separator,
            comment_prefix,
            quote_char,
            eol_char,
            missing_is_null,
            ignore_errors,
            truncate_ragged_lines,
            null_values,
            projection,
            &mut buffers,
            chunk_size,
            schema.len(),
            schema,
        )?;
    }

    let columns = buffers
        .into_iter()
        .map(|buf| buf.into_series())
        .collect::<PolarsResult<_>>()?;
    Ok(unsafe { DataFrame::new_no_checks(columns) })
}
