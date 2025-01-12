pub(super) mod batched;

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, handle_casting_failures};
use polars_core::POOL;
#[cfg(feature = "polars-time")]
use polars_time::prelude::*;
use rayon::prelude::*;

use super::buffer::init_buffers;
use super::options::{CommentPrefix, CsvEncoding, NullValuesCompiled};
use super::parser::{
    is_comment_line, parse_lines, skip_bom, skip_line_ending, skip_lines_naive, skip_this_line,
    CountLines, SplitLines,
};
use super::reader::prepare_csv_schema;
use super::schema_inference::{check_decimal_comma, infer_file_schema};
#[cfg(feature = "decompress")]
use super::utils::decompress;
use super::CsvParseOptions;
use crate::csv::read::parser::skip_this_line_naive;
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
use crate::utils::compression::SupportedCompression;
use crate::utils::update_row_counts2;
use crate::RowIndex;

pub fn cast_columns(
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
        *df = unsafe { DataFrame::new_no_checks(df.height(), cols) }
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
    parse_options: CsvParseOptions,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// Current line number, used in error reporting
    current_line: usize,
    ignore_errors: bool,
    skip_lines: usize,
    skip_rows_before_header: usize,
    // after the header, we need to take embedded lines into account
    skip_rows_after_header: usize,
    n_rows: Option<usize>,
    n_threads: Option<usize>,
    has_header: bool,
    chunk_size: usize,
    null_values: Option<NullValuesCompiled>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    to_cast: Vec<Field>,
    row_index: Option<RowIndex>,
    #[cfg_attr(not(feature = "dtype-categorical"), allow(unused))]
    has_categorical: bool,
}

impl fmt::Debug for CoreReader<'_> {
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
        parse_options: Arc<CsvParseOptions>,
        n_rows: Option<usize>,
        skip_rows: usize,
        skip_lines: usize,
        mut projection: Option<Vec<usize>>,
        max_records: Option<usize>,
        has_header: bool,
        ignore_errors: bool,
        schema: Option<SchemaRef>,
        columns: Option<Arc<[PlSmallStr]>>,
        mut n_threads: Option<usize>,
        schema_overwrite: Option<SchemaRef>,
        dtype_overwrite: Option<Arc<Vec<DataType>>>,
        chunk_size: usize,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        mut to_cast: Vec<Field>,
        skip_rows_after_header: usize,
        row_index: Option<RowIndex>,
        raise_if_empty: bool,
    ) -> PolarsResult<CoreReader<'a>> {
        let separator = parse_options.separator;

        check_decimal_comma(parse_options.decimal_comma, separator)?;
        #[cfg(feature = "decompress")]
        let mut reader_bytes = reader_bytes;

        if !cfg!(feature = "decompress") && SupportedCompression::check(&reader_bytes).is_some() {
            polars_bail!(
                ComputeError: "cannot read compressed CSV file; \
                compile with feature 'decompress'"
            );
        }
        // We keep track of the inferred schema bool
        // In case the file is compressed this schema inference is wrong and has to be done
        // again after decompression.
        #[cfg(feature = "decompress")]
        {
            let total_n_rows =
                n_rows.map(|n| skip_rows + (has_header as usize) + skip_rows_after_header + n);
            if let Some(b) = decompress(
                &reader_bytes,
                total_n_rows,
                separator,
                parse_options.quote_char,
                parse_options.eol_char,
            ) {
                reader_bytes = ReaderBytes::Owned(b.into());
            }
        }

        let mut schema = match schema {
            Some(schema) => schema,
            None => {
                let (inferred_schema, _, _) = infer_file_schema(
                    &reader_bytes,
                    &parse_options,
                    max_records,
                    has_header,
                    schema_overwrite.as_deref(),
                    skip_rows,
                    skip_lines,
                    skip_rows_after_header,
                    raise_if_empty,
                    &mut n_threads,
                )?;
                Arc::new(inferred_schema)
            },
        };
        if let Some(dtypes) = dtype_overwrite {
            polars_ensure!(
                dtypes.len() <= schema.len(),
                InvalidOperation: "The number of schema overrides must be less than or equal to the number of fields"
            );
            let s = Arc::make_mut(&mut schema);
            for (index, dt) in dtypes.iter().enumerate() {
                s.set_dtype_at_index(index, dt.clone()).unwrap();
            }
        }

        let has_categorical = prepare_csv_schema(&mut schema, &mut to_cast)?;

        // Create a null value for every column
        let null_values = parse_options
            .null_values
            .as_ref()
            .map(|nv| nv.clone().compile(&schema))
            .transpose()?;

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
            parse_options: (*parse_options).clone(),
            schema,
            projection,
            current_line: usize::from(has_header),
            ignore_errors,
            skip_lines,
            skip_rows_before_header: skip_rows,
            skip_rows_after_header,
            n_rows,
            n_threads,
            has_header,
            chunk_size,
            null_values,
            predicate,
            to_cast,
            row_index,
            has_categorical,
        })
    }

    fn find_starting_point<'b>(
        &self,
        bytes: &'b [u8],
        quote_char: Option<u8>,
        eol_char: u8,
    ) -> PolarsResult<(&'b [u8], Option<usize>)> {
        let i = find_starting_point(
            bytes,
            quote_char,
            eol_char,
            self.schema.len(),
            self.skip_lines,
            self.skip_rows_before_header,
            self.skip_rows_after_header,
            self.parse_options.comment_prefix.as_ref(),
            self.has_header,
        )?;

        Ok((&bytes[i..], (i <= bytes.len()).then_some(i)))
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
            &self.parse_options,
            self.schema.as_ref(),
            self.ignore_errors,
            projection,
            bytes_offset,
            capacity,
            self.null_values.as_ref(),
            usize::MAX,
            stop_at_nbytes,
            starting_point_offset,
        )?;

        cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;
        Ok(df)
    }

    fn parse_csv(&mut self, bytes: &[u8]) -> PolarsResult<DataFrame> {
        let (bytes, _) = self.find_starting_point(
            bytes,
            self.parse_options.quote_char,
            self.parse_options.eol_char,
        )?;

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

        let counter = CountLines::new(self.parse_options.quote_char, self.parse_options.eol_char);
        let mut total_offset = 0;
        let check_utf8 = matches!(self.parse_options.encoding, CsvEncoding::Utf8)
            && self.schema.iter_fields().any(|f| f.dtype().is_string());

        pool.scope(|s| {
            loop {
                let b = unsafe { bytes.get_unchecked(total_offset..) };
                if b.is_empty() {
                    break;
                }
                debug_assert!(
                    total_offset == 0 || bytes[total_offset - 1] == self.parse_options.eol_char
                );
                let (count, position) = counter.find_next(b, &mut chunk_size);
                debug_assert!(count == 0 || b[position] == self.parse_options.eol_char);

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

                    let end = total_offset + position + 1;
                    let b = unsafe { bytes.get_unchecked(total_offset..end) };

                    total_offset = end;
                    (b, count)
                };

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
                            .read_chunk(b, projection, 0, count, Some(0), b.len())
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
    pub fn finish(mut self) -> PolarsResult<DataFrame> {
        #[cfg(feature = "dtype-categorical")]
        let mut _cat_lock = if self.has_categorical {
            Some(polars_core::StringCacheHolder::hold())
        } else {
            None
        };

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
pub fn read_chunk(
    bytes: &[u8],
    parse_options: &CsvParseOptions,
    schema: &Schema,
    ignore_errors: bool,
    projection: &[usize],
    bytes_offset_thread: usize,
    capacity: usize,
    null_values: Option<&NullValuesCompiled>,
    chunk_size: usize,
    stop_at_nbytes: usize,
    starting_point_offset: Option<usize>,
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
        parse_options.quote_char,
        parse_options.encoding,
        parse_options.decimal_comma,
    )?;

    debug_assert!(projection.is_sorted());

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
            parse_options,
            offset,
            ignore_errors,
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
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(unsafe { DataFrame::new_no_checks_height_from_first(columns) })
}

#[allow(clippy::too_many_arguments)]
pub fn find_starting_point(
    mut bytes: &[u8],
    quote_char: Option<u8>,
    eol_char: u8,
    schema_len: usize,
    skip_lines: usize,
    skip_rows_before_header: usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    has_header: bool,
) -> PolarsResult<usize> {
    let full_len = bytes.len();
    let starting_point_offset = bytes.as_ptr() as usize;

    bytes = if skip_lines > 0 {
        polars_ensure!(skip_rows_before_header == 0, InvalidOperation: "only one of 'skip_rows'/'skip_lines' may be set");
        skip_lines_naive(bytes, eol_char, skip_lines)
    } else {
        // Skip utf8 byte-order-mark (BOM)
        bytes = skip_bom(bytes);

        // \n\n can be a empty string row of a single column
        // in other cases we skip it.
        if schema_len > 1 {
            bytes = skip_line_ending(bytes, eol_char)
        }
        bytes
    };

    // skip 'n' leading rows
    if skip_rows_before_header > 0 {
        let mut split_lines = SplitLines::new(bytes, quote_char, eol_char, comment_prefix);
        let mut current_line = &bytes[..0];

        for _ in 0..skip_rows_before_header {
            current_line = split_lines
                .next()
                .ok_or_else(|| polars_err!(NoData: "not enough lines to skip"))?;
        }

        current_line = split_lines
            .next()
            .unwrap_or(&current_line[current_line.len()..]);
        bytes = &bytes[current_line.as_ptr() as usize - bytes.as_ptr() as usize..];
    }

    // skip lines that are comments
    while is_comment_line(bytes, comment_prefix) {
        bytes = skip_this_line_naive(bytes, eol_char);
    }

    // skip header row
    if has_header {
        bytes = skip_this_line(bytes, quote_char, eol_char);
    }
    // skip 'n' rows following the header
    if skip_rows_after_header > 0 {
        let mut split_lines = SplitLines::new(bytes, quote_char, eol_char, comment_prefix);
        let mut current_line = &bytes[..0];

        for _ in 0..skip_rows_after_header {
            current_line = split_lines
                .next()
                .ok_or_else(|| polars_err!(NoData: "not enough lines to skip"))?;
        }

        current_line = split_lines
            .next()
            .unwrap_or(&current_line[current_line.len()..]);
        bytes = &bytes[current_line.as_ptr() as usize - bytes.as_ptr() as usize..];
    }

    Ok(
        // Some of the functions we call may return `&'static []` instead of
        // slices of `&bytes[..]`.
        if bytes.is_empty() {
            full_len
        } else {
            bytes.as_ptr() as usize - starting_point_offset
        },
    )
}
