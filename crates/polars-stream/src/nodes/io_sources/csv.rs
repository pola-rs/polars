use std::cmp;
use std::iter::Iterator;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use polars_core::prelude::{Field, Schema};
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err, polars_warn};
use polars_io::cloud::CloudOptions;
use polars_io::prelude::_csv_read_internal::{
    CommentPrefix, CountLines, NullValuesCompiled, SplitLines, cast_columns, is_comment_line,
    prepare_csv_schema, read_chunk,
};
use polars_io::prelude::buffer::validate_utf8;
use polars_io::prelude::{CsvEncoding, CsvParseOptions, CsvReadOptions};
use polars_io::utils::compression::{CompressedReader, maybe_decompress_bytes};
use polars_io::utils::slice::SplitSlicePosition;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mmap::MemSlice;
use polars_utils::slice_enum::Slice;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, FileReader, FileReaderCallbacks};
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::distributor_channel::{self, distributor_channel};
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::{MorselSeq, TaskPriority};

pub mod builder {
    use std::sync::Arc;

    use polars_core::config;
    use polars_io::cloud::CloudOptions;
    use polars_io::prelude::CsvReadOptions;
    use polars_plan::dsl::ScanSource;

    use super::CsvFileReader;
    use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
    use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
    use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

    impl FileReaderBuilder for Arc<CsvReadOptions> {
        fn reader_name(&self) -> &str {
            "csv"
        }

        fn reader_capabilities(&self) -> ReaderCapabilities {
            use ReaderCapabilities as RC;

            RC::NEEDS_FILE_CACHE_INIT
                | if self.parse_options.comment_prefix.is_some() {
                    RC::empty()
                } else {
                    RC::PRE_SLICE
                }
        }

        fn build_file_reader(
            &self,
            source: ScanSource,
            cloud_options: Option<Arc<CloudOptions>>,
            _scan_source_idx: usize,
        ) -> Box<dyn FileReader> {
            let scan_source = source;
            let verbose = config::verbose();
            let options = self.clone();

            let reader = CsvFileReader {
                scan_source,
                cloud_options,
                options,
                verbose,
                cached_bytes: None,
            };

            Box::new(reader) as Box<dyn FileReader>
        }
    }
}

/// Read all rows in the chunk
const NO_SLICE: (usize, usize) = (0, usize::MAX);
/// This is used if we finish the slice but still need a row count. It signals to the workers to
/// go into line-counting mode where they can skip parsing the chunks.
const SLICE_ENDED: (usize, usize) = (usize::MAX, 0);

struct LineBatch {
    // Safety: All receivers (LineBatchProcessors) hold a MemSlice ref to this.
    mem_slice: MemSlice,
    n_lines: usize,
    slice: (usize, usize),
    /// Position of this chunk relative to the start of the file according to CountLines.
    row_offset: usize,
    morsel_seq: MorselSeq,
}

struct CsvFileReader {
    scan_source: ScanSource,
    #[expect(unused)] // Will be used when implementing cloud streaming.
    cloud_options: Option<Arc<CloudOptions>>,
    options: Arc<CsvReadOptions>,
    // Cached on first access - we may be called multiple times e.g. on negative slice.
    cached_bytes: Option<MemSlice>,
    verbose: bool,
}

#[async_trait]
impl FileReader for CsvFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        let memslice = self
            .scan_source
            .as_scan_source_ref()
            .to_memslice_async_assume_latest(self.scan_source.run_async())?;

        // Note: We do not decompress in `initialize()`.
        self.cached_bytes = Some(memslice);

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let reader = CompressedReader::try_new(self.cached_bytes.clone().unwrap())?;

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            // Because we currently only support PRE_SLICE we don't need to handle row index here.
            row_index,
            pre_slice,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        assert!(row_index.is_none()); // Handled outside the reader for now.

        match &pre_slice {
            Some(Slice::Negative { .. }) => unimplemented!(),

            // We don't account for comments when slicing lines. We should never hit this panic -
            // the FileReaderBuilder does not indicate PRE_SLICE support when we have a comment
            // prefix.
            Some(pre_slice)
                if self.options.parse_options.comment_prefix.is_some() && pre_slice.len() > 0 =>
            {
                panic!("{pre_slice:?}")
            },

            _ => {},
        }

        // TODO: Always compare inferred and provided schema once schema inference can handle
        // streaming decompression.
        let used_schema = if let Some(schema) = &self.options.schema
            && !self.options.parse_options.truncate_ragged_lines
        {
            schema.clone()
        } else {
            Arc::new(self.infer_schema(projected_schema.clone())?)
        };

        if let Some(tx) = file_schema_tx {
            _ = tx.send(used_schema.clone())
        }

        let projection: Vec<usize> = projected_schema
            .iter_names()
            .filter_map(|name| used_schema.index_of(name))
            .collect();

        if verbose {
            eprintln!(
                "[CsvFileReader]: project: {} / {}, slice: {:?}",
                projection.len(),
                used_schema.len(),
                &pre_slice,
            )
        }

        let quote_char = self.options.parse_options.quote_char;
        let eol_char = self.options.parse_options.eol_char;
        let comment_prefix = self.options.parse_options.comment_prefix.clone();

        let line_counter = CountLines::new(quote_char, eol_char, comment_prefix.clone());

        let chunk_reader = Arc::new(ChunkReader::try_new(
            self.options.clone(),
            used_schema.clone(),
            projection,
        )?);

        let needs_full_row_count = n_rows_in_file_tx.is_some();

        let (line_batch_tx, line_batch_receivers) =
            distributor_channel(num_pipelines, *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let line_batch_source_handle = AbortOnDropHandle::new(spawn(
            TaskPriority::Low,
            LineBatchSource {
                reader,
                line_counter,
                line_batch_tx,
                options: self.options.clone(),
                file_schema_len: used_schema.len(),
                pre_slice,
                needs_full_row_count,
                verbose,
            }
            .run(),
        ));

        let n_workers = line_batch_receivers.len();

        let (morsel_senders, rx) = FileReaderOutputSend::new_parallel(num_pipelines);

        let line_batch_decode_handles = line_batch_receivers
            .into_iter()
            .zip(morsel_senders)
            .enumerate()
            .map(|(worker_idx, (mut line_batch_rx, mut morsel_tx))| {
                // Only verbose log from the last worker to avoid flooding output.
                let verbose = verbose && worker_idx == n_workers - 1;
                let mut n_rows_processed: usize = 0;
                let chunk_reader = chunk_reader.clone();
                // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
                let source_token = SourceToken::new();

                AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
                    while let Ok(LineBatch {
                        mem_slice,
                        n_lines,
                        slice,
                        row_offset,
                        morsel_seq,
                    }) = line_batch_rx.recv().await
                    {
                        let (offset, len) = match slice {
                            SLICE_ENDED => (0, 1),
                            v => v,
                        };

                        let (df, n_rows_in_chunk) = chunk_reader.read_chunk(
                            &mem_slice,
                            n_lines,
                            (offset, len),
                            row_offset,
                        )?;

                        n_rows_processed = n_rows_processed.saturating_add(n_rows_in_chunk);

                        if (offset, len) == SLICE_ENDED {
                            break;
                        }

                        let morsel = Morsel::new(df, morsel_seq, source_token.clone());

                        if morsel_tx.send_morsel(morsel).await.is_err() {
                            break;
                        }
                    }

                    drop(morsel_tx);

                    if needs_full_row_count {
                        if verbose {
                            eprintln!(
                                "[CSV LineBatchProcessor {worker_idx}]: entering row count mode"
                            );
                        }

                        while let Ok(LineBatch {
                            mem_slice: _,
                            n_lines,
                            slice,
                            row_offset: _,
                            morsel_seq: _,
                        }) = line_batch_rx.recv().await
                        {
                            assert_eq!(slice, SLICE_ENDED);

                            n_rows_processed = n_rows_processed.saturating_add(n_lines);
                        }
                    }

                    PolarsResult::Ok(n_rows_processed)
                }))
            })
            .collect::<Vec<_>>();

        Ok((
            rx,
            spawn(TaskPriority::Low, async move {
                let mut row_position: usize = 0;

                for handle in line_batch_decode_handles {
                    let rows_processed = handle.await?;
                    row_position = row_position.saturating_add(rows_processed);
                }

                row_position = {
                    let rows_skipped = line_batch_source_handle.await?;
                    row_position.saturating_add(rows_skipped)
                };

                let row_position = IdxSize::try_from(row_position)
                    .map_err(|_| polars_err!(bigidx, ctx = "csv file", size = row_position))?;

                if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                    assert!(needs_full_row_count);
                    _ = n_rows_in_file_tx.send(row_position);
                }

                if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                    _ = row_position_on_end_tx.send(row_position);
                }

                Ok(())
            }),
        ))
    }
}

impl CsvFileReader {
    /// Does *not* handle compressed files in a streaming manner.
    fn infer_schema(&self, projected_schema: SchemaRef) -> PolarsResult<Schema> {
        // We need to infer the schema to get the columns of this file.
        let infer_schema_length = if self.options.has_header {
            Some(1)
        } else {
            // If there is no header the line length may increase later in the
            // file (https://github.com/pola-rs/polars/pull/21979).
            self.options.infer_schema_length
        };

        let mut decompress_buf = Vec::new();
        let bytes =
            maybe_decompress_bytes(self.cached_bytes.as_ref().unwrap(), &mut decompress_buf)?;

        let (mut inferred_schema, ..) = polars_io::csv::read::infer_file_schema(
            &polars_io::mmap::ReaderBytes::Borrowed(bytes),
            &self.options.parse_options,
            infer_schema_length,
            self.options.has_header,
            self.options.schema_overwrite.as_deref(),
            self.options.skip_rows,
            self.options.skip_lines,
            self.options.skip_rows_after_header,
            self.options.raise_if_empty,
        )?;

        if let Some(schema) = &self.options.schema {
            // Note: User can provide schema with more columns, they will simply
            // be projected as NULL.
            // TODO: Should maybe expose a missing_columns parameter to the API for this.
            if schema.len() < inferred_schema.len()
                && !self.options.parse_options.truncate_ragged_lines
            {
                polars_bail!(
                    SchemaMismatch:
                    "provided schema does not match number of columns in file ({} != {} in file)",
                    schema.len(),
                    inferred_schema.len(),
                );
            }

            if self.options.parse_options.truncate_ragged_lines {
                inferred_schema = Arc::unwrap_or_clone(schema.clone());
            } else {
                inferred_schema = schema
                    .iter_names()
                    .zip(inferred_schema.into_iter().map(|(_, dtype)| dtype))
                    .map(|(name, dtype)| (name.clone(), dtype))
                    .collect();
            }
        }

        if let Some(dtypes) = self.options.dtype_overwrite.as_deref() {
            for (i, dtype) in dtypes.iter().enumerate() {
                inferred_schema.set_dtype_at_index(i, dtype.clone());
            }
        }

        // TODO
        // We currently always override with the projected dtype, but this may cause
        // issues e.g. with temporal types. This can be improved to better choose
        // between the 2 dtypes.
        for (name, inferred_dtype) in inferred_schema.iter_mut() {
            if let Some(projected_dtype) = projected_schema.get(name) {
                *inferred_dtype = projected_dtype.clone();
            }
        }

        Ok(inferred_schema)
    }
}

struct LineBatchSource {
    reader: CompressedReader,
    line_counter: CountLines,
    line_batch_tx: distributor_channel::Sender<LineBatch>,
    options: Arc<CsvReadOptions>,
    file_schema_len: usize,
    pre_slice: Option<Slice>,
    needs_full_row_count: bool,
    verbose: bool,
}

impl LineBatchSource {
    /// Returns the number of rows skipped from the start of the file according to CountLines.
    async fn run(self) -> PolarsResult<usize> {
        let LineBatchSource {
            mut reader,
            line_counter,
            mut line_batch_tx,
            options,
            file_schema_len,
            pre_slice,
            needs_full_row_count,
            verbose,
        } = self;

        let global_slice = if let Some(pre_slice) = pre_slice {
            match pre_slice {
                Slice::Positive { .. } => Some(Range::<usize>::from(pre_slice)),
                // IR lowering puts negative slice in separate node.
                // TODO: Native line buffering for negative slice
                Slice::Negative { .. } => unreachable!(),
            }
        } else {
            None
        };

        if verbose {
            eprintln!("[CsvSource]: Start line splitting",);
        }

        let parse_options = options.parse_options.as_ref();

        let mut prev_leftover = read_until_starting_point(
            parse_options.quote_char,
            parse_options.eol_char,
            file_schema_len,
            options.skip_lines,
            options.skip_rows,
            options.skip_rows_after_header,
            options.parse_options.comment_prefix.as_ref(),
            options.has_header,
            options.raise_if_empty,
            &mut reader,
        )?;

        let mut row_offset = 0usize;
        let mut morsel_seq = MorselSeq::default();
        let mut n_rows_skipped: usize = 0;
        let mut read_size = 512 * 1024; // L2 sized chunks performed the best in testing.

        loop {
            let (mem_slice, bytes_read) = reader.read_next_slice(&prev_leftover, read_size)?;
            if mem_slice.is_empty() {
                break;
            }

            mem_slice.prefetch();

            let is_eof = bytes_read == 0;
            let (n_lines, unconsumed_offset) = line_counter.count_rows(&mem_slice, is_eof);

            let batch_slice = mem_slice.slice(0..unconsumed_offset);
            prev_leftover = mem_slice.slice(unconsumed_offset..mem_slice.len());

            if batch_slice.is_empty() && !is_eof {
                // This allows the slice to grow until at least a single row is included. To avoid a quadratic run-time for large row sizes, we double the read size.
                read_size *= 2;
                continue;
            }

            // Has to happen here before slicing, since there are slice operations that skip morsel
            // sending.
            let prev_row_offset = row_offset;
            row_offset += n_lines;

            let slice = if let Some(global_slice) = &global_slice {
                match SplitSlicePosition::split_slice_at_file(
                    prev_row_offset,
                    n_lines,
                    global_slice.clone(),
                ) {
                    // Note that we don't check that the skipped line batches actually contain this many
                    // lines.
                    SplitSlicePosition::Before => {
                        n_rows_skipped = n_rows_skipped.saturating_add(n_lines);
                        continue;
                    },
                    SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    SplitSlicePosition::After => {
                        if needs_full_row_count {
                            // If we need to know the unrestricted row count, we need
                            // to go until the end.
                            SLICE_ENDED
                        } else {
                            break;
                        }
                    },
                }
            } else {
                NO_SLICE
            };

            morsel_seq = morsel_seq.successor();

            let batch = LineBatch {
                mem_slice: batch_slice,
                n_lines,
                slice,
                row_offset,
                morsel_seq,
            };

            if line_batch_tx.send(batch).await.is_err() {
                break;
            }

            if is_eof {
                break;
            }
        }

        Ok(n_rows_skipped)
    }
}

#[derive(Default)]
struct ChunkReader {
    reader_schema: SchemaRef,
    parse_options: Arc<CsvParseOptions>,
    fields_to_cast: Vec<Field>,
    ignore_errors: bool,
    projection: Vec<usize>,
    null_values: Option<NullValuesCompiled>,
    validate_utf8: bool,
}

impl ChunkReader {
    fn try_new(
        options: Arc<CsvReadOptions>,
        mut reader_schema: SchemaRef,
        projection: Vec<usize>,
    ) -> PolarsResult<Self> {
        let mut fields_to_cast: Vec<Field> = options.fields_to_cast.clone();
        prepare_csv_schema(&mut reader_schema, &mut fields_to_cast)?;

        let parse_options = options.parse_options.clone();

        // Logic from `CoreReader::new()`

        let null_values = parse_options
            .null_values
            .clone()
            .map(|nv| nv.compile(&reader_schema))
            .transpose()?;

        let validate_utf8 = matches!(parse_options.encoding, CsvEncoding::Utf8)
            && reader_schema.iter_fields().any(|f| f.dtype().is_string());

        Ok(Self {
            reader_schema,
            parse_options,
            fields_to_cast,
            ignore_errors: options.ignore_errors,
            projection,
            null_values,
            validate_utf8,
        })
    }

    /// The 2nd return value indicates how many rows exist in the chunk.
    fn read_chunk(
        &self,
        chunk: &[u8],
        // Number of lines according to CountLines
        n_lines: usize,
        slice: (usize, usize),
        chunk_row_offset: usize,
    ) -> PolarsResult<(DataFrame, usize)> {
        if self.validate_utf8 && !validate_utf8(chunk) {
            polars_bail!(ComputeError: "invalid utf-8 sequence")
        }

        // If projection is empty create a DataFrame with the correct height by counting the lines.
        let mut df = if self.projection.is_empty() {
            DataFrame::empty_with_height(n_lines)
        } else {
            read_chunk(
                chunk,
                &self.parse_options,
                &self.reader_schema,
                self.ignore_errors,
                &self.projection,
                0,       // bytes_offset_thread
                n_lines, // capacity
                self.null_values.as_ref(),
                usize::MAX,  // chunk_size
                chunk.len(), // stop_at_nbytes
                Some(0),     // starting_point_offset
            )?
        };

        let height = df.height();

        if height != n_lines {
            // Note: in case data is malformed, height is more likely to be correct than n_lines.
            let msg = format!(
                "CSV malformed: expected {} rows, actual {} rows, in chunk starting at row_offset {}, length {}",
                n_lines,
                height,
                chunk_row_offset,
                chunk.len()
            );
            if self.ignore_errors {
                polars_warn!("{}", msg);
            } else {
                polars_bail!(ComputeError: msg);
            }
        }

        if slice != NO_SLICE {
            assert!(slice != SLICE_ENDED);

            df = df.slice(i64::try_from(slice.0).unwrap(), slice.1);
        }

        cast_columns(&mut df, &self.fields_to_cast, false, self.ignore_errors)?;

        Ok((df, height))
    }
}

/// Reads bytes from `reader` until the CSV starting point is reached depending on the options.
///
/// Returns the leftover bytes not yet consumed, which may be empty.
///
/// The reading is done in an iterative streaming fashion
#[expect(clippy::too_many_arguments)]
pub fn read_until_starting_point(
    quote_char: Option<u8>,
    eol_char: u8,
    schema_len: usize,
    skip_lines: usize,
    skip_rows_before_header: usize,
    skip_rows_after_header: usize,
    comment_prefix: Option<&CommentPrefix>,
    has_header: bool,
    raise_if_empty: bool,
    reader: &mut CompressedReader,
) -> PolarsResult<MemSlice> {
    #[derive(Copy, Clone)]
    enum State {
        // Ordered so that all states only happen after the ones before it.
        SkipEmpty,
        SkipRowsBeforeHeader(usize),
        SkipHeader(bool),
        SkipRowsAfterHeader(usize),
        Done,
    }

    polars_ensure!(
        !(skip_lines != 0 && skip_rows_before_header != 0),
        InvalidOperation: "only one of 'skip_rows'/'skip_lines' may be set"
    );

    // We have to treat skip_lines differently since the lines it skips may not follow regular CSV
    // quote escape rules.
    let prev_leftover = skip_lines_naive(eol_char, skip_lines, raise_if_empty, reader)?;

    let mut state = if schema_len > 1 || has_header {
        State::SkipEmpty
    } else if skip_lines != 0 {
        // skip_lines shouldn't skip extra comments before the header, so directly go to SkipHeader
        // state.
        State::SkipHeader(false)
    } else {
        State::SkipRowsBeforeHeader(skip_rows_before_header)
    };

    for_each_line_from_reader(
        quote_char,
        eol_char,
        comment_prefix,
        true,
        prev_leftover,
        reader,
        |line| {
            let done = loop {
                match &mut state {
                    State::SkipEmpty => {
                        if line.is_empty() || line == b"\r" {
                            break false;
                        }

                        state = State::SkipRowsBeforeHeader(skip_rows_before_header);
                    },
                    State::SkipRowsBeforeHeader(remaining) => {
                        let is_comment = is_comment_line(line, comment_prefix);

                        if *remaining == 0 && !is_comment {
                            state = State::SkipHeader(false);
                            continue;
                        }

                        *remaining -= !is_comment as usize;
                        break false;
                    },
                    State::SkipHeader(did_skip) => {
                        if !has_header || *did_skip {
                            state = State::SkipRowsAfterHeader(skip_rows_after_header);
                            continue;
                        }

                        *did_skip = true;
                        break false;
                    },
                    State::SkipRowsAfterHeader(remaining) => {
                        let is_comment = is_comment_line(line, comment_prefix);

                        if *remaining == 0 && !is_comment {
                            state = State::Done;
                            continue;
                        }

                        *remaining -= !is_comment as usize;
                        break false;
                    },
                    State::Done => {
                        break true;
                    },
                }
            };

            Ok(done)
        },
    )
}

/// Iterate over valid CSV lines produced by reader.
pub fn for_each_line_from_reader(
    quote_char: Option<u8>,
    eol_char: u8,
    comment_prefix: Option<&CommentPrefix>,
    is_file_start: bool,
    mut prev_leftover: MemSlice,
    reader: &mut CompressedReader,
    mut line_fn: impl FnMut(&[u8]) -> PolarsResult<bool>,
) -> PolarsResult<MemSlice> {
    let mut is_first_line = is_file_start;

    // Since this is used for schema inference, we want to avoid needlessly large reads at first.
    let mut read_size = 128 * 1024;

    loop {
        let (mut slice, bytes_read) = reader.read_next_slice(&prev_leftover, read_size)?;
        if slice.is_empty() {
            return Ok(MemSlice::EMPTY);
        }

        if is_first_line {
            is_first_line = false;
            const UTF8_BOM_MARKER: Option<&[u8]> = Some(b"\xef\xbb\xbf");
            if slice.get(0..3) == UTF8_BOM_MARKER {
                slice = slice.slice(3..slice.len());
            }
        }

        let mut lines = SplitLines::new(&slice, quote_char, eol_char, comment_prefix);
        let Some(mut prev_line) = lines.next() else {
            read_size *= 2;
            prev_leftover = slice;
            continue;
        };

        let mut should_ret = false;

        // The last line in `SplitLines` may be incomplete if `slice` ends before the file does, so
        // we iterate everything except the last line.
        for next_line in lines {
            if line_fn(prev_line)? {
                should_ret = true;
                break;
            }
            prev_line = next_line;
        }

        // EOF file reached, the last line will have no continuation on the next call to
        // `read_next_slice`.
        if bytes_read == 0 {
            if !line_fn(prev_line)? {
                // The `line_fn` wants to consume more lines, but there aren't any.
                // Make sure to report to the caller that there is no leftover.
                prev_line = &slice[slice.len()..slice.len()];
            }
            should_ret = true;
        }

        let unconsumed_offset = prev_line.as_ptr() as usize - slice.as_ptr() as usize;
        prev_leftover = slice.slice(unconsumed_offset..slice.len());

        if should_ret {
            return Ok(prev_leftover);
        }
    }
}

fn skip_lines_naive(
    eol_char: u8,
    skip_lines: usize,
    raise_if_empty: bool,
    reader: &mut CompressedReader,
) -> PolarsResult<MemSlice> {
    let mut prev_leftover = MemSlice::EMPTY;

    if skip_lines == 0 {
        return Ok(prev_leftover);
    }

    let mut remaining = skip_lines;
    // Since this is used for schema inference, we want to avoid needlessly large reads at first.
    let mut read_size = 128 * 1024;

    loop {
        let (slice, bytes_read) = reader.read_next_slice(&prev_leftover, read_size)?;
        let mut bytes: &[u8] = &slice;

        'inner: loop {
            let Some(mut pos) = memchr::memchr(eol_char, bytes) else {
                read_size *= 2;
                break 'inner;
            };
            pos = cmp::min(pos + 1, bytes.len());

            bytes = &bytes[pos..];
            remaining -= 1;

            if remaining == 0 {
                let unconsumed_offset = bytes.as_ptr() as usize - slice.as_ptr() as usize;
                prev_leftover = slice.slice(unconsumed_offset..slice.len());
                return Ok(prev_leftover);
            }
        }

        if bytes_read == 0 {
            if raise_if_empty {
                polars_bail!(NoData: "specified skip_lines is larger than total number of lines.");
            }
            // Return empty slice to signal no data remaining
            return Ok(MemSlice::EMPTY);
        }

        // No need to search for naive eol twice in the leftover.
        prev_leftover = MemSlice::EMPTY;
    }
}
