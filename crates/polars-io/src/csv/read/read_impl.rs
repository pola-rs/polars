pub(super) mod batched;

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, handle_casting_failures};
use polars_core::POOL;
#[cfg(feature = "polars-time")]
use polars_time::prelude::*;
use polars_utils::slice::GetSaferUnchecked;
use rayon::prelude::*;

use super::buffer::init_buffers;
use super::options::{CommentPrefix, CsvEncoding, NullValues, NullValuesCompiled};
use super::parser::{
    is_comment_line, next_line_position, next_line_position_naive, parse_lines, skip_bom,
    skip_line_ending, skip_this_line, CountLines,
};
use super::schema_inference::{check_decimal_comma, infer_file_schema};
#[cfg(any(feature = "decompress", feature = "decompress-fast"))]
use super::utils::decompress;
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
#[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
use crate::utils::compression::SupportedCompression;
use crate::utils::update_row_counts2;
use crate::RowIndex;

pub(crate) fn cast_columns(
    df: &mut DataFrame,
    to_cast: &[Field],
    parallel: bool,
    ignore_errors: bool,
) -> PolarsResult<()> {
    let cast_fn = |c: &Column, fld: &Field| {
        let out = match (c.dtype(), fld.dtype()) {
            #[cfg(feature = "temporal")]
            (DataType::String, DataType::Date) => c
                .str()
                .unwrap()
                .as_date(None, false)
                .map(|ca| ca.into_column()),
            #[cfg(feature = "temporal")]
            (DataType::String, DataType::Time) => c
                .str()
                .unwrap()
                .as_time(None, false)
                .map(|ca| ca.into_column()),
            #[cfg(feature = "temporal")]
            (DataType::String, DataType::Datetime(tu, _)) => c
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
                .map(|ca| ca.into_column()),
            (_, dt) => c.cast(dt),
        }?;
        if !ignore_errors && c.null_count() != out.null_count() {
            handle_casting_failures(c.as_materialized_series(), out.as_materialized_series())?;
        }
        Ok(out)
    };

    if parallel {
        let cols = POOL.install(|| {
            df.get_columns()
                .into_par_iter()
                .map(|s| {
                    if let Some(fld) = to_cast.iter().find(|fld| fld.name() == s.name()) {
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
    chunk_size: usize,
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
        columns: Option<Arc<[PlSmallStr]>>,
        encoding: CsvEncoding,
        mut n_threads: Option<usize>,
        schema_overwrite: Option<SchemaRef>,
        dtype_overwrite: Option<Arc<Vec<DataType>>>,
        chunk_size: usize,
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
        if SupportedCompression::check(&reader_bytes).is_some() {
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
            chunk_size,
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

        // Skip utf8 byte-order-mark (BOM)
        bytes = skip_bom(bytes);

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

    fn read_chunk(
        &self,
        bytes: &[u8],
        projection: &[usize],
        bytes_offset: usize,
        capacity: usize,
        starting_point_offset: Option<usize>,
        stop_at_nbytes: usize,
    ) -> PolarsResult<DataFrame> {
        let mut df = read_chunk(
            bytes,
            self.separator,
            self.schema.as_ref(),
            self.ignore_errors,
            projection,
            bytes_offset,
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
        Ok(df)
    }

    fn parse_csv(&mut self, bytes: &[u8]) -> PolarsResult<DataFrame> {
        let (bytes, starting_point_offset) =
            self.find_starting_point(bytes, self.quote_char, self.eol_char)?;

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
                df.insert_column(0, Series::new_empty(row_index.name.clone(), &IDX_DTYPE))?;
            }
            return Ok(df);
        }

        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        // This is chosen by benchmarking on ny city trip csv dataset.
        // We want small enough chunks such that threads start working as soon as possible
        // But we also want them large enough, so that we have less chunks related overhead, but
        // We minimize chunks to 16 MB to still fit L3 cache.
        let n_parts_hint = n_threads * 16;
        let chunk_size = std::cmp::min(bytes.len() / n_parts_hint, 16 * 1024 * 1024);

        // Use a small min chunk size to catch failures in tests.
        #[cfg(debug_assertions)]
        let min_chunk_size = 64;
        #[cfg(not(debug_assertions))]
        let min_chunk_size = 1024 * 4;

        let mut chunk_size = std::cmp::max(chunk_size, min_chunk_size);
        let mut total_bytes_offset = 0;

        let results = Arc::new(Mutex::new(vec![]));
        // We have to do this after parsing as there can be comments.
        let total_line_count = &AtomicUsize::new(0);

        #[cfg(not(target_family = "wasm"))]
        let pool;
        #[cfg(not(target_family = "wasm"))]
        let pool = if n_threads == POOL.current_num_threads() {
            &POOL
        } else {
            pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|_| polars_err!(ComputeError: "could not spawn threads"))?;
            &pool
        };
        #[cfg(target_family = "wasm")]
        let pool = &POOL;

        let counter = CountLines::new(self.quote_char, self.eol_char);
        let mut total_offset = 0;

        pool.scope(|s| {
            loop {
                let b = unsafe {
                    bytes.get_unchecked_release(
                        total_offset..std::cmp::min(total_offset + chunk_size, bytes.len()),
                    )
                };
                if b.is_empty() {
                    break;
                }
                debug_assert!(total_offset == 0 || bytes[total_offset - 1] == self.eol_char);
                let (count, position) = counter.count(b);
                debug_assert!(count == 0 || b[position] == self.eol_char);

                let (b, count) = if count == 0
                    && unsafe { b.as_ptr().add(b.len()) == bytes.as_ptr().add(bytes.len()) }
                {
                    total_offset = bytes.len();
                    (b, 1)
                } else {
                    if count == 0 {
                        chunk_size *= 2;
                        continue;
                    }

                    let b = unsafe {
                        bytes.get_unchecked_release(total_offset..total_offset + position)
                    };
                    // The parsers will not create a null row if we end on a new line.
                    if b.last() == Some(&self.eol_char) {
                        chunk_size *= 2;
                        continue;
                    }

                    total_offset += position + 1;
                    (b, count)
                };
                let check_utf8 = matches!(self.encoding, CsvEncoding::Utf8)
                    && self.schema.iter_fields().any(|f| f.dtype().is_string());

                if !b.is_empty() {
                    let results = results.clone();
                    let projection = projection.as_ref();
                    let slf = &(*self);
                    s.spawn(move |_| {
                        if check_utf8 && !super::buffer::validate_utf8(b) {
                            let mut results = results.lock().unwrap();
                            results.push((
                                b.as_ptr() as usize,
                                Err(polars_err!(ComputeError: "invalid utf-8 sequence")),
                            ));
                            return;
                        }

                        let result = slf
                            .read_chunk(b, projection, 0, count, starting_point_offset, b.len())
                            .and_then(|mut df| {
                                debug_assert!(df.height() <= count);

                                if slf.n_rows.is_some() {
                                    total_line_count.fetch_add(df.height(), Ordering::Relaxed);
                                }

                                // We cannot use the line count as there can be comments in the lines so we must correct line counts later.
                                if let Some(rc) = &slf.row_index {
                                    // is first chunk
                                    let offset = if b.as_ptr() == bytes.as_ptr() {
                                        Some(rc.offset)
                                    } else {
                                        None
                                    };

                                    df.with_row_index_mut(rc.name.clone(), offset);
                                };

                                if let Some(predicate) = slf.predicate.as_ref() {
                                    let s = predicate.evaluate_io(&df)?;
                                    let mask = s.bool()?;
                                    df = df.filter(mask)?;
                                }
                                Ok(df)
                            });

                        results.lock().unwrap().push((b.as_ptr() as usize, result));
                    });

                    // Check just after we spawned a chunk. That mean we processed all data up until
                    // row count.
                    if self.n_rows.is_some()
                        && total_line_count.load(Ordering::Relaxed) > self.n_rows.unwrap()
                    {
                        break;
                    }
                }
                total_bytes_offset += b.len();
            }
        });
        let mut results = std::mem::take(&mut *results.lock().unwrap());
        results.sort_unstable_by_key(|k| k.0);
        let mut dfs = results
            .into_iter()
            .map(|k| k.1)
            .collect::<PolarsResult<Vec<_>>>()?;

        if let Some(rc) = &self.row_index {
            update_row_counts2(&mut dfs, rc.offset)
        };
        accumulate_dataframes_vertical(dfs)
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(&mut self) -> PolarsResult<DataFrame> {
        let reader_bytes = self.reader_bytes.take().unwrap();

        let mut df = self.parse_csv(&reader_bytes)?;

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
        .map(|buf| buf.into_series().map(Column::from))
        .collect::<PolarsResult<_>>()?;
    Ok(unsafe { DataFrame::new_no_checks(columns) })
}
