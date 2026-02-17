use std::fmt;
use std::sync::Mutex;

use polars_buffer::{Buffer, SharedStorage};
use polars_core::POOL;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, handle_casting_failures};
#[cfg(feature = "polars-time")]
use polars_time::prelude::*;
use polars_utils::relaxed_cell::RelaxedCell;
use rayon::prelude::*;

use super::CsvParseOptions;
use super::builder::init_builders;
use super::options::{CsvEncoding, NullValuesCompiled};
use super::parser::{CountLines, is_comment_line, parse_lines};
use super::reader::prepare_csv_schema;
#[cfg(feature = "decompress")]
use super::utils::decompress;
use crate::RowIndex;
use crate::csv::read::{CsvReadOptions, read_until_start_and_infer_schema};
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;
use crate::utils::compression::{CompressedReader, SupportedCompression};
use crate::utils::update_row_counts2;

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
            df.columns()
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
        *df = unsafe { DataFrame::new_unchecked(df.height(), cols) }
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

struct ReaderBytesAndDependents<'a> {
    // Ensure lifetime dependents are dropped before `reader_bytes`, since their drop impls
    // could access themselves, this is achieved by placing them before `reader_bytes`.
    // SAFETY: This is lifetime bound to `reader_bytes`
    compressed_reader: CompressedReader,
    // SAFETY: This is lifetime bound to `reader_bytes`
    leftover: Buffer<u8>,
    _reader_bytes: ReaderBytes<'a>,
}

/// CSV file reader
pub(crate) struct CoreReader<'a> {
    reader_bytes: Option<ReaderBytesAndDependents<'a>>,

    /// Explicit schema for the CSV file
    schema: SchemaRef,
    parse_options: CsvParseOptions,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// Current line number, used in error reporting
    current_line: usize,
    ignore_errors: bool,
    n_rows: Option<usize>,
    n_threads: Option<usize>,
    null_values: Option<NullValuesCompiled>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    to_cast: Vec<Field>,
    row_index: Option<RowIndex>,
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
        n_threads: Option<usize>,
        schema_overwrite: Option<SchemaRef>,
        dtype_overwrite: Option<Arc<Vec<DataType>>>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        mut to_cast: Vec<Field>,
        skip_rows_after_header: usize,
        row_index: Option<RowIndex>,
        raise_if_empty: bool,
    ) -> PolarsResult<CoreReader<'a>> {
        let separator = parse_options.separator;

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

        let reader_slice = match &reader_bytes {
            ReaderBytes::Borrowed(slice) => {
                // SAFETY: The produced slice and derived slices MUST not live longer than
                // `reader_bytes`. TODO use `scan_csv` to implement `read_csv`.
                let ss = unsafe { SharedStorage::from_slice_unchecked(slice) };
                Buffer::from_storage(ss)
            },
            ReaderBytes::Owned(slice) => slice.clone(),
        };
        let mut compressed_reader = CompressedReader::try_new(reader_slice)?;

        let read_options = CsvReadOptions {
            parse_options: parse_options.clone(),
            n_rows,
            skip_rows,
            skip_lines,
            projection: projection.clone().map(Arc::new),
            has_header,
            ignore_errors,
            schema: schema.clone(),
            columns: columns.clone(),
            n_threads,
            schema_overwrite,
            dtype_overwrite: dtype_overwrite.clone(),
            fields_to_cast: to_cast.clone(),
            skip_rows_after_header,
            row_index: row_index.clone(),
            raise_if_empty,
            infer_schema_length: max_records,
            ..Default::default()
        };

        // Since this is also used to skip to the start, always call it.
        let (inferred_schema, leftover) =
            read_until_start_and_infer_schema(&read_options, None, None, &mut compressed_reader)?;

        let mut schema = match schema {
            Some(schema) => schema,
            None => Arc::new(inferred_schema),
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

        prepare_csv_schema(&mut schema, &mut to_cast)?;

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
            reader_bytes: Some(ReaderBytesAndDependents {
                compressed_reader,
                leftover,
                _reader_bytes: reader_bytes,
            }),
            parse_options: (*parse_options).clone(),
            schema,
            projection,
            current_line: usize::from(has_header),
            ignore_errors,
            n_rows,
            n_threads,
            null_values,
            predicate,
            to_cast,
            row_index,
        })
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

    // The code adheres to RFC 4180 in a strict sense, unless explicitly documented otherwise.
    // Malformed CSV is common, see e.g. the use of lazy_quotes, whitespace and comments.
    // In case malformed CSV is detected, a warning or an error will be issued.
    // Not all malformed CSV will be detected, as that would impact performance.
    fn parse_csv(&mut self, bytes: &[u8]) -> PolarsResult<DataFrame> {
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

            cast_columns(&mut df, &self.to_cast, false, self.ignore_errors)?;

            if let Some(ref row_index) = self.row_index {
                df.insert_column(0, Column::new_empty(row_index.name.clone(), &IDX_DTYPE))?;
            }
            return Ok(df);
        }

        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        // This is chosen by benchmarking on ny city trip csv dataset.
        // We want small enough chunks such that threads start working as soon as possible
        // But we also want them large enough, so that we have less chunks related overhead.
        // We minimize chunks to 16 MB to still fit L3 cache.
        //
        // Width-aware adjustment: For wide data (many columns), per-chunk overhead
        // (allocating column buffers) becomes significant. Each chunk must allocate
        // O(n_cols) buffers, so total allocation overhead is O(n_chunks * n_cols).
        // To keep this bounded, we limit n_chunks such that n_chunks * n_cols <= threshold.
        // With threshold ~500K, this gives:
        //   - 100 cols: up to 5000 chunks (no practical limit)
        //   - 1000 cols: up to 500 chunks
        //   - 10000 cols: up to 50 chunks
        //   - 30000 cols: up to 16 chunks
        let n_cols = projection.len();
        // Empirically determined to balance allocation overhead and parallelism.
        const ALLOCATION_BUDGET: usize = 500_000;
        let max_chunks_for_width = ALLOCATION_BUDGET / n_cols.max(1);
        let n_parts_hint = std::cmp::min(n_threads * 16, max_chunks_for_width.max(n_threads));
        let chunk_size = std::cmp::min(bytes.len() / n_parts_hint.max(1), 16 * 1024 * 1024);

        // Use a small min chunk size to catch failures in tests.
        #[cfg(debug_assertions)]
        let min_chunk_size = 64;
        #[cfg(not(debug_assertions))]
        let min_chunk_size = 1024 * 4;

        let mut chunk_size = std::cmp::max(chunk_size, min_chunk_size);
        let mut total_bytes_offset = 0;

        let results = Arc::new(Mutex::new(vec![]));
        // We have to do this after parsing as there can be comments.
        let total_line_count = &RelaxedCell::new_usize(0);

        let counter = CountLines::new(
            self.parse_options.quote_char,
            self.parse_options.eol_char,
            None,
        );
        let mut total_offset = 0;
        let mut previous_total_offset = 0;
        let check_utf8 = matches!(self.parse_options.encoding, CsvEncoding::Utf8)
            && self.schema.iter_fields().any(|f| f.dtype().is_string());

        POOL.scope(|s| {
            // Pass 1: identify chunks for parallel processing (line parsing).
            loop {
                let b = unsafe { bytes.get_unchecked(total_offset..) };
                if b.is_empty() {
                    break;
                }
                debug_assert!(
                    total_offset == 0 || bytes[total_offset - 1] == self.parse_options.eol_char
                );

                // Count is the number of rows for the next chunk. In case of malformed CSV data,
                // count may not be as expected.
                let (count, position) = counter.find_next(b, &mut chunk_size);
                debug_assert!(count == 0 || b[position] == self.parse_options.eol_char);

                let (b, count) = if count == 0
                    && unsafe {
                        std::ptr::eq(b.as_ptr().add(b.len()), bytes.as_ptr().add(bytes.len()))
                    } {
                    total_offset = bytes.len();
                    let c = if is_comment_line(bytes, self.parse_options.comment_prefix.as_ref()) {
                        0
                    } else {
                        1
                    };
                    (b, c)
                } else {
                    let end = total_offset + position + 1;
                    let b = unsafe { bytes.get_unchecked(total_offset..end) };

                    previous_total_offset = total_offset;
                    total_offset = end;
                    (b, count)
                };

                // Pass 2: process each individual chunk in parallel (field parsing)
                if !b.is_empty() {
                    let results = results.clone();
                    let projection = projection.as_ref();
                    let slf = &(*self);
                    s.spawn(move |_| {
                        if check_utf8 && !super::builder::validate_utf8(b) {
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
                                // Check malformed
                                if df.height() > count
                                    || (df.height() < count
                                        && slf.parse_options.comment_prefix.is_none())
                                {
                                    // Note: in case data is malformed, df.height() is more likely to be correct than count.
                                    let msg = format!(
                                        "CSV malformed: expected {} rows, \
                                        actual {} rows, in chunk starting at \
                                        byte offset {}, length {}",
                                        count,
                                        df.height(),
                                        previous_total_offset,
                                        b.len()
                                    );
                                    if slf.ignore_errors {
                                        polars_warn!("{msg}");
                                    } else {
                                        polars_bail!(ComputeError: msg)
                                    }
                                }

                                if slf.n_rows.is_some() {
                                    total_line_count.fetch_add(df.height());
                                }

                                // We cannot use the line count as there can be comments in the lines so we must correct line counts later.
                                if let Some(rc) = &slf.row_index {
                                    // is first chunk
                                    let offset = if std::ptr::eq(b.as_ptr(), bytes.as_ptr()) {
                                        Some(rc.offset)
                                    } else {
                                        None
                                    };

                                    unsafe { df.with_row_index_mut(rc.name.clone(), offset) };
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
                    if self.n_rows.is_some() && total_line_count.load() > self.n_rows.unwrap() {
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
        let mut reader_bytes = self.reader_bytes.take().unwrap();
        let (body_bytes, _) = reader_bytes
            .compressed_reader
            .read_next_slice(&reader_bytes.leftover, usize::MAX)?;

        let mut df = self.parse_csv(&body_bytes)?;

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
    let mut buffers = init_builders(
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
    Ok(unsafe { DataFrame::new_unchecked_infer_height(columns) })
}
