use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
#[cfg(feature = "dtype-categorical")]
use polars_core::StringCacheHolder;
use polars_core::prelude::{Column, Field};
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_io::prelude::_csv_read_internal::{
    CountLines, NullValuesCompiled, cast_columns, find_starting_point, prepare_csv_schema,
    read_chunk,
};
use polars_io::prelude::buffer::validate_utf8;
use polars_io::prelude::{
    CommentPrefix, CsvEncoding, CsvParseOptions, CsvReadOptions, count_rows_from_slice,
};
use polars_io::utils::compression::maybe_decompress_bytes;
use polars_io::utils::slice::SplitSlicePosition;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mmap::MemSlice;
use polars_utils::slice_enum::Slice;

use super::multi_file_reader::reader_interface::output::FileReaderOutputRecv;
use super::multi_file_reader::reader_interface::{BeginReadArgs, FileReader, FileReaderCallbacks};
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::distributor_channel::{self, distributor_channel};
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputSend;
use crate::nodes::{MorselSeq, TaskPriority};

pub mod builder {
    use std::sync::Arc;

    use polars_core::config;
    use polars_io::cloud::CloudOptions;
    use polars_io::prelude::CsvReadOptions;
    use polars_plan::dsl::ScanSource;

    use super::CsvFileReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;

    impl FileReaderBuilder for Arc<CsvReadOptions> {
        fn reader_name(&self) -> &str {
            "csv"
        }

        fn reader_capabilities(&self) -> ReaderCapabilities {
            use ReaderCapabilities as RC;

            if self.parse_options.comment_prefix.is_some() {
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
    bytes: &'static [u8],
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

        let memslice = self.get_bytes_maybe_decompress()?;

        let BeginReadArgs {
            projected_schema,
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

        match &pre_slice {
            // We don't account for comments when slicing lines. We should never hit this panic -
            // the FileReaderBuilder does not indicate PRE_SLICE support when we have a comment
            // prefix.
            Some(..) if self.options.parse_options.comment_prefix.is_some() => panic!(),
            Some(Slice::Negative { .. }) => unimplemented!(),
            _ => {},
        }

        // We need to infer the schema to get the columns of this file.
        let infer_schema_length = if self.options.has_header {
            Some(1)
        } else {
            // If there is no header the line length may increase later in the
            // file (https://github.com/pola-rs/polars/pull/21979).
            self.options.infer_schema_length
        };

        let (mut inferred_schema, ..) = polars_io::csv::read::infer_file_schema(
            &polars_io::mmap::ReaderBytes::Owned(memslice.clone()),
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

        let inferred_schema = Arc::new(inferred_schema);

        if let Some(mut tx) = file_schema_tx {
            _ = tx.try_send(inferred_schema.clone())
        }

        let projection: Vec<usize> = projected_schema
            .iter_names()
            .filter_map(|name| inferred_schema.index_of(name))
            .collect();

        if verbose {
            eprintln!(
                "[CsvFileReader]: project: {} / {}, slice: {:?}, row_index: {:?}",
                projection.len(),
                inferred_schema.len(),
                &pre_slice,
                row_index,
            )
        }

        // Only used on empty projection, or if we need the exact row count.
        let alt_count_lines: Option<Arc<CountLinesWithComments>> =
            CountLinesWithComments::opt_new(&self.options.parse_options).map(Arc::new);
        let chunk_reader = Arc::new(ChunkReader::try_new(
            self.options.clone(),
            inferred_schema.clone(),
            projection,
            row_index,
            alt_count_lines.clone(),
        )?);

        let needs_full_row_count = n_rows_in_file_tx.is_some();

        let (line_batch_tx, line_batch_receivers) =
            distributor_channel(num_pipelines, *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let line_batch_source_handle = AbortOnDropHandle::new(spawn(
            TaskPriority::Low,
            LineBatchSource {
                memslice: memslice.clone(),
                line_counter: CountLines::new(
                    self.options.parse_options.quote_char,
                    self.options.parse_options.eol_char,
                ),
                line_batch_tx,
                options: self.options.clone(),
                file_schema_len: inferred_schema.len(),
                pre_slice,
                needs_full_row_count,
                num_pipelines,
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
                // Hold a ref as we are receiving `&'static [u8]`s pointing to this.
                let global_memslice = memslice.clone();
                // Only verbose log from the last worker to avoid flooding output.
                let verbose = verbose && worker_idx == n_workers - 1;
                let mut n_rows_processed: usize = 0;
                let chunk_reader = chunk_reader.clone();
                // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
                let source_token = SourceToken::new();
                let alt_count_lines = alt_count_lines.clone();

                AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
                    while let Ok(LineBatch {
                        bytes,
                        n_lines,
                        slice,
                        row_offset,
                        morsel_seq,
                    }) = line_batch_rx.recv().await
                    {
                        debug_assert!(bytes.as_ptr() as usize >= global_memslice.as_ptr() as usize);
                        debug_assert!(
                            bytes.as_ptr() as usize + bytes.len()
                                <= global_memslice.as_ptr() as usize + global_memslice.len()
                        );

                        let (offset, len) = match slice {
                            SLICE_ENDED => (0, 1),
                            v => v,
                        };

                        let (df, n_rows_in_chunk) =
                            chunk_reader.read_chunk(bytes, n_lines, (offset, len), row_offset)?;

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
                                "[CSV LineBatchProcessor {}]: entering row count mode",
                                worker_idx
                            );
                        }

                        while let Ok(LineBatch {
                            bytes,
                            n_lines,
                            slice,
                            row_offset: _,
                            morsel_seq: _,
                        }) = line_batch_rx.recv().await
                        {
                            assert_eq!(slice, SLICE_ENDED);

                            let n_lines = if let Some(v) = alt_count_lines.as_deref() {
                                v.count_lines(bytes)?
                            } else {
                                n_lines
                            };

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

                if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
                    assert!(needs_full_row_count);
                    _ = n_rows_in_file_tx.try_send(row_position);
                }

                if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
                    _ = row_position_on_end_tx.try_send(row_position);
                }

                Ok(())
            }),
        ))
    }
}

impl CsvFileReader {
    /// # Panics
    /// Panics if `self.cached_bytes` is None.
    fn get_bytes_maybe_decompress(&mut self) -> PolarsResult<MemSlice> {
        let mut out = vec![];
        maybe_decompress_bytes(self.cached_bytes.as_deref().unwrap(), &mut out)?;

        if !out.is_empty() {
            self.cached_bytes = Some(MemSlice::from_vec(out));
        }

        Ok(self.cached_bytes.clone().unwrap())
    }
}

struct LineBatchSource {
    memslice: MemSlice,
    line_counter: CountLines,
    line_batch_tx: distributor_channel::Sender<LineBatch>,
    options: Arc<CsvReadOptions>,
    file_schema_len: usize,
    pre_slice: Option<Slice>,
    needs_full_row_count: bool,
    num_pipelines: usize,
    verbose: bool,
}

impl LineBatchSource {
    /// Returns the number of rows skipped from the start of the file according to CountLines.
    async fn run(self) -> PolarsResult<usize> {
        let LineBatchSource {
            memslice,
            line_counter,
            mut line_batch_tx,
            options,
            file_schema_len,
            pre_slice,
            needs_full_row_count,
            num_pipelines,
            verbose,
        } = self;

        let mut n_rows_skipped: usize = 0;

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

        let morsel_seq_ref = &mut MorselSeq::default();
        let current_row_offset_ref = &mut 0usize;

        if verbose {
            eprintln!("[CsvSource]: Start line splitting",);
        }

        let global_bytes: &[u8] = memslice.as_ref();
        let global_bytes: &'static [u8] = unsafe { std::mem::transmute(global_bytes) };

        let i = {
            let parse_options = options.parse_options.as_ref();

            let quote_char = parse_options.quote_char;
            let eol_char = parse_options.eol_char;

            let skip_lines = options.skip_lines;
            let skip_rows_before_header = options.skip_rows;
            let skip_rows_after_header = options.skip_rows_after_header;
            let comment_prefix = parse_options.comment_prefix.clone();
            let has_header = options.has_header;

            find_starting_point(
                global_bytes,
                quote_char,
                eol_char,
                file_schema_len,
                skip_lines,
                skip_rows_before_header,
                skip_rows_after_header,
                comment_prefix.as_ref(),
                has_header,
            )?
        };

        let mut bytes = &global_bytes[i..];

        let mut chunk_size = {
            let max_chunk_size = 16 * 1024 * 1024;
            let chunk_size = if global_slice.is_some() {
                max_chunk_size
            } else {
                std::cmp::min(bytes.len() / (16 * num_pipelines), max_chunk_size)
            };

            // Use a small min chunk size to catch failures in tests.
            #[cfg(debug_assertions)]
            let min_chunk_size = 64;
            #[cfg(not(debug_assertions))]
            let min_chunk_size = 1024 * 4;
            std::cmp::max(chunk_size, min_chunk_size)
        };

        loop {
            if bytes.is_empty() {
                break;
            }

            let (count, position) = line_counter.find_next(bytes, &mut chunk_size);
            let (count, position) = if count == 0 {
                (1, bytes.len())
            } else {
                let pos = (position + 1).min(bytes.len()); // +1 for '\n'
                (count, pos)
            };

            let slice_start = bytes.as_ptr() as usize - global_bytes.as_ptr() as usize;

            bytes = &bytes[position..];

            let current_row_offset = *current_row_offset_ref;
            *current_row_offset_ref += count;

            let slice = if let Some(global_slice) = &global_slice {
                match SplitSlicePosition::split_slice_at_file(
                    current_row_offset,
                    count,
                    global_slice.clone(),
                ) {
                    // Note that we don't check that the skipped line batches actually contain this many
                    // lines.
                    SplitSlicePosition::Before => {
                        n_rows_skipped = n_rows_skipped.saturating_add(count);
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

            let bytes_this_chunk = &global_bytes[slice_start..slice_start + position];

            let morsel_seq = *morsel_seq_ref;
            *morsel_seq_ref = morsel_seq.successor();

            let batch = LineBatch {
                bytes: bytes_this_chunk,
                n_lines: count,
                slice,
                row_offset: current_row_offset,
                morsel_seq,
            };

            if line_batch_tx.send(batch).await.is_err() {
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
    #[cfg(feature = "dtype-categorical")]
    _cat_lock: Option<StringCacheHolder>,
    ignore_errors: bool,
    projection: Vec<usize>,
    null_values: Option<NullValuesCompiled>,
    validate_utf8: bool,
    row_index: Option<RowIndex>,
    // Alternate line counter when there are comments. This is used on empty projection.
    alt_count_lines: Option<Arc<CountLinesWithComments>>,
}

impl ChunkReader {
    fn try_new(
        options: Arc<CsvReadOptions>,
        mut reader_schema: SchemaRef,
        projection: Vec<usize>,
        row_index: Option<RowIndex>,
        alt_count_lines: Option<Arc<CountLinesWithComments>>,
    ) -> PolarsResult<Self> {
        let mut fields_to_cast: Vec<Field> = options.fields_to_cast.clone();
        let has_categorical = prepare_csv_schema(&mut reader_schema, &mut fields_to_cast)?;

        #[cfg(feature = "dtype-categorical")]
        let _cat_lock = has_categorical.then(polars_core::StringCacheHolder::hold);

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
            #[cfg(feature = "dtype-categorical")]
            _cat_lock,
            ignore_errors: options.ignore_errors,
            projection,
            null_values,
            validate_utf8,
            row_index,
            alt_count_lines,
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
            let h = if let Some(v) = &self.alt_count_lines {
                v.count_lines(chunk)?
            } else {
                n_lines
            };

            DataFrame::empty_with_height(h)
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
        let n_lines_is_correct = df.height() == n_lines;

        if slice != NO_SLICE {
            assert!(slice != SLICE_ENDED);
            assert!(n_lines_is_correct);

            df = df.slice(i64::try_from(slice.0).unwrap(), slice.1);
        }

        cast_columns(&mut df, &self.fields_to_cast, false, self.ignore_errors)?;

        if let Some(ri) = &self.row_index {
            assert!(n_lines_is_correct);

            unsafe {
                df.with_column_unchecked(Column::new_row_index(
                    ri.name.clone(),
                    ri.offset
                        .saturating_add(chunk_row_offset.try_into().unwrap_or(IdxSize::MAX)),
                    df.height(),
                )?);
            }
        }

        Ok((df, height))
    }
}

struct CountLinesWithComments {
    quote_char: Option<u8>,
    eol_char: u8,
    comment_prefix: CommentPrefix,
}

impl CountLinesWithComments {
    fn opt_new(parse_options: &CsvParseOptions) -> Option<Self> {
        parse_options
            .comment_prefix
            .clone()
            .map(|comment_prefix| CountLinesWithComments {
                quote_char: parse_options.quote_char,
                eol_char: parse_options.eol_char,
                comment_prefix,
            })
    }

    fn count_lines(&self, bytes: &[u8]) -> PolarsResult<usize> {
        count_rows_from_slice(
            bytes,
            self.quote_char,
            Some(&self.comment_prefix),
            self.eol_char,
            false, // has_header
        )
    }
}
