use std::sync::atomic::Ordering;
use std::sync::Arc;

use polars_core::config;
use polars_core::prelude::Field;
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_core::utils::arrow::bitmap::Bitmap;
#[cfg(feature = "dtype-categorical")]
use polars_core::StringCacheHolder;
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_io::cloud::CloudOptions;
use polars_io::prelude::_csv_read_internal::{
    cast_columns, find_starting_point, prepare_csv_schema, read_chunk, CountLines,
    NullValuesCompiled,
};
use polars_io::prelude::buffer::validate_utf8;
use polars_io::prelude::{CsvEncoding, CsvParseOptions, CsvReadOptions};
use polars_io::utils::compression::maybe_decompress_bytes;
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::RowIndex;
use polars_plan::plans::{isolated_csv_file_info, FileInfo, ScanSource};
use polars_plan::prelude::FileScanOptions;
use polars_utils::index::AtomicIdxSize;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use super::multi_scan::{MultiScanable, RowRestrication};
use super::{SourceNode, SourceOutput};
use crate::async_executor::{self, spawn};
use crate::async_primitives::connector::{connector, Receiver};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::MorselOutput;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;

struct LineBatch {
    bytes: MemSlice,
    n_lines: usize,
    slice: (usize, usize),
    row_offset: usize,
    morsel_seq: MorselSeq,
}

type AsyncTaskData = (
    Vec<crate::async_primitives::distributor_channel::Receiver<LineBatch>>,
    Arc<ChunkReader>,
    async_executor::AbortOnDropHandle<PolarsResult<()>>,
);

pub struct CsvSourceNode {
    scan_source: ScanSource,
    file_info: FileInfo,
    file_options: FileScanOptions,
    options: CsvReadOptions,
    schema: Option<SchemaRef>,
    verbose: bool,
}

impl CsvSourceNode {
    pub fn new(
        scan_source: ScanSource,
        file_info: FileInfo,
        file_options: FileScanOptions,
        options: CsvReadOptions,
    ) -> Self {
        let verbose = config::verbose();

        Self {
            scan_source,
            file_info,
            file_options,
            options,
            schema: None,
            verbose,
        }
    }
}

impl SourceNode for CsvSourceNode {
    fn name(&self) -> &str {
        "csv_source"
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
        true
    }

    fn spawn_source(
        &mut self,
        num_pipelines: usize,
        mut output_recv: Receiver<SourceOutput>,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<Arc<AtomicIdxSize>>,
    ) {
        let (mut send_to, recv_from) = (0..num_pipelines)
            .map(|_| connector::<MorselOutput>())
            .collect::<(Vec<_>, Vec<_>)>();

        self.schema = Some(self.file_info.reader_schema.take().unwrap().unwrap_right());

        let source_token = SourceToken::new();
        let (line_batch_receivers, chunk_reader, line_batch_source_task_handle) =
            self.init_line_batch_source(num_pipelines, unrestricted_row_count);

        join_handles.extend(line_batch_receivers.into_iter().zip(recv_from).map(
            |(mut line_batch_rx, mut recv_from)| {
                let chunk_reader = chunk_reader.clone();
                let source_token = source_token.clone();
                let wait_group = WaitGroup::default();

                spawn(TaskPriority::Low, async move {
                    while let Ok(mut morsel_output) = recv_from.recv().await {
                        while let Ok(LineBatch {
                            bytes,
                            n_lines,
                            slice: (offset, len),
                            row_offset,
                            morsel_seq,
                        }) = line_batch_rx.recv().await
                        {
                            let df = chunk_reader.read_chunk(
                                &bytes,
                                n_lines,
                                (offset, len),
                                row_offset,
                            )?;

                            let mut morsel = Morsel::new(df, morsel_seq, source_token.clone());
                            morsel.set_consume_token(wait_group.token());

                            if morsel_output.port.send(morsel).await.is_err() {
                                break;
                            }
                            wait_group.wait().await;

                            if source_token.stop_requested() {
                                morsel_output.outcome.stop();
                                break;
                            }
                        }
                    }

                    PolarsResult::Ok(())
                })
            },
        ));

        join_handles.push(spawn(TaskPriority::Low, async move {
            // Every phase we are given a new send port.
            while let Ok(phase_output) = output_recv.recv().await {
                let morsel_senders = phase_output.port.parallel();
                let mut morsel_outcomes = Vec::with_capacity(morsel_senders.len());

                for (send_to, port) in send_to.iter_mut().zip(morsel_senders) {
                    let (outcome, wait_group, morsel_output) = MorselOutput::from_port(port);
                    _ = send_to.send(morsel_output).await;
                    morsel_outcomes.push((outcome, wait_group));
                }

                let mut is_finished = true;
                for (outcome, wait_group) in morsel_outcomes.into_iter() {
                    wait_group.wait().await;
                    is_finished &= outcome.did_finish();
                }

                if is_finished {
                    break;
                }

                phase_output.outcome.stop();
            }

            drop(send_to);
            // Join on the producer handle to catch errors/panics.
            // Safety
            // * We dropped the receivers on the line above
            // * This function is only called once.
            line_batch_source_task_handle.await
        }))
    }
}

impl CsvSourceNode {
    fn init_line_batch_source(
        &mut self,
        num_pipelines: usize,
        unrestricted_row_count: Option<Arc<AtomicIdxSize>>,
    ) -> AsyncTaskData {
        let verbose = self.verbose;

        let (mut line_batch_sender, line_batch_receivers) =
            distributor_channel(num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let scan_source = self.scan_source.clone();
        let run_async = matches!(&scan_source, ScanSource::Path(p) if polars_io::is_cloud_url(p) || config::force_async());

        let schema_len = self.schema.as_ref().unwrap().len();

        let options = &self.options;
        let parse_options = self.options.parse_options.as_ref();

        let quote_char = parse_options.quote_char;
        let eol_char = parse_options.eol_char;

        let skip_lines = options.skip_lines;
        let skip_rows_before_header = options.skip_rows;
        let skip_rows_after_header = options.skip_rows_after_header;
        let comment_prefix = parse_options.comment_prefix.clone();
        let has_header = options.has_header;
        let global_slice = self.file_options.slice;

        if verbose {
            eprintln!(
                "[CsvSource]: slice: {:?}, row_index: {:?}",
                global_slice, &self.file_options.row_index
            )
        }

        if global_slice.is_some() {
            assert!(comment_prefix.is_none()) // We don't account for comments when slicing lines.
        }

        // This function doesn't return a Result type, so we send Option<Err> into the task and
        // propagate it from there instead to avoid `unwrap()` panicking.
        let chunk_reader = self.try_init_chunk_reader();
        let chunk_reader_init_err = chunk_reader
            .as_ref()
            .map_err(|e| e.wrap_msg(|x| format!("csv_source::ChunkReader init error: {}", x)))
            .err();

        let line_batch_source_task_handle = async_executor::AbortOnDropHandle::new(
            async_executor::spawn(TaskPriority::Low, async move {
                let global_slice = if let Some((offset, len)) = global_slice {
                    if offset < 0 {
                        polars_bail!(
                            ComputeError:
                            "not implemented: negative slice offset {} for CSV source",
                            offset
                        );
                    }
                    Some(offset as usize..offset as usize + len)
                } else {
                    None
                };

                if let Some(err) = chunk_reader_init_err {
                    return Err(err);
                }

                let line_counter = CountLines::new(quote_char, eol_char);

                let morsel_seq_ref = &mut MorselSeq::default();
                let current_row_offset_ref = &mut 0usize;
                let mem_slice = scan_source
                    .as_scan_source_ref()
                    .to_memslice_async_assume_latest(run_async)?;

                if verbose {
                    eprintln!("[CsvSource]: Start line splitting",);
                }

                let mem_slice = {
                    let mut out = vec![];
                    maybe_decompress_bytes(&mem_slice, &mut out)?;

                    if out.is_empty() {
                        mem_slice
                    } else {
                        MemSlice::from_vec(out)
                    }
                };

                let bytes = mem_slice.as_ref();

                let i = find_starting_point(
                    bytes,
                    quote_char,
                    eol_char,
                    schema_len,
                    skip_lines,
                    skip_rows_before_header,
                    skip_rows_after_header,
                    comment_prefix.as_ref(),
                    has_header,
                )?;

                let mut bytes = &bytes[i..];

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

                    let slice_start = bytes.as_ptr() as usize - mem_slice.as_ptr() as usize;

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
                            SplitSlicePosition::Before => continue,
                            SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                            SplitSlicePosition::After => {
                                if unrestricted_row_count.is_some() {
                                    // If we need to know the unrestricted row count, we need
                                    // to go until the end.
                                    continue;
                                } else {
                                    break;
                                }
                            },
                        }
                    } else {
                        // (0, 0) is interpreted as no slicing
                        (0, 0)
                    };

                    let mem_slice_this_chunk = mem_slice.slice(slice_start..slice_start + position);

                    let morsel_seq = *morsel_seq_ref;
                    *morsel_seq_ref = morsel_seq.successor();

                    let batch = LineBatch {
                        bytes: mem_slice_this_chunk,
                        n_lines: count,
                        slice,
                        row_offset: current_row_offset,
                        morsel_seq,
                    };
                    if line_batch_sender.send(batch).await.is_err() {
                        break;
                    }
                }

                if let Some(unrestricted_row_count) = unrestricted_row_count.as_ref() {
                    let num_rows = *current_row_offset_ref;
                    let num_rows = IdxSize::try_from(num_rows)
                        .map_err(|_| polars_err!(bigidx, ctx = "csv file", size = num_rows))?;
                    unrestricted_row_count.store(num_rows, Ordering::Relaxed);
                }

                Ok(())
            }),
        );

        (
            line_batch_receivers,
            Arc::new(chunk_reader.unwrap_or_default()),
            line_batch_source_task_handle,
        )
    }

    fn try_init_chunk_reader(&mut self) -> PolarsResult<ChunkReader> {
        let with_columns = self
            .file_options
            .with_columns
            .clone()
            // Interpret selecting no columns as selecting all columns.
            .filter(|columns| !columns.is_empty());

        ChunkReader::try_new(
            &mut self.options,
            self.schema.as_ref().unwrap(),
            with_columns.as_deref(),
            self.file_options.row_index.clone(),
        )
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
}

impl ChunkReader {
    fn try_new(
        options: &mut CsvReadOptions,
        reader_schema: &SchemaRef,
        with_columns: Option<&[PlSmallStr]>,
        row_index: Option<RowIndex>,
    ) -> PolarsResult<Self> {
        let mut reader_schema = reader_schema.clone();
        // Logic from `CsvReader::finish()`
        let mut fields_to_cast = std::mem::take(&mut options.fields_to_cast);

        if let Some(dtypes) = options.dtype_overwrite.as_deref() {
            let mut s = Arc::unwrap_or_clone(reader_schema);
            for (i, dtype) in dtypes.iter().enumerate() {
                s.set_dtype_at_index(i, dtype.clone());
            }
            reader_schema = s.into();
        }

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

        let projection = if let Some(cols) = with_columns {
            let mut v = Vec::with_capacity(cols.len());
            for col in cols {
                v.push(reader_schema.try_index_of(col)?);
            }
            v.sort_unstable();
            v
        } else if let Some(v) = options.projection.clone() {
            let mut v = Arc::unwrap_or_clone(v);
            v.sort_unstable();
            v
        } else {
            (0..reader_schema.len()).collect::<Vec<_>>()
        };

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
        })
    }

    fn read_chunk(
        &self,
        chunk: &[u8],
        n_lines: usize,
        slice: (usize, usize),
        chunk_row_offset: usize,
    ) -> PolarsResult<DataFrame> {
        if self.validate_utf8 && !validate_utf8(chunk) {
            polars_bail!(ComputeError: "invalid utf-8 sequence")
        }

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
        )
        .and_then(|mut df| {
            let n_lines_is_correct = df.height() == n_lines;

            if slice != (0, 0) {
                assert!(n_lines_is_correct);

                df = df.slice(slice.0 as i64, slice.1);
            }

            cast_columns(&mut df, &self.fields_to_cast, false, self.ignore_errors)?;

            if let Some(ri) = &self.row_index {
                assert!(n_lines_is_correct);

                let offset = ri.offset;

                let Some(offset) = (|| {
                    let offset = offset.checked_add((chunk_row_offset + slice.0) as IdxSize)?;
                    offset.checked_add(df.height() as IdxSize)?;

                    Some(offset)
                })() else {
                    let msg = format!(
                        "adding a row index column with offset {} overflows at {} rows",
                        offset,
                        chunk_row_offset + slice.0
                    );
                    polars_bail!(ComputeError: msg)
                };

                df.with_row_index_mut(ri.name.clone(), Some(offset as IdxSize));
            }

            Ok(df)
        })
    }
}

impl MultiScanable for CsvSourceNode {
    type ReadOptions = CsvReadOptions;

    const BASE_NAME: &'static str = "csv";

    const DOES_PRED_PD: bool = false;
    const DOES_SLICE_PD: bool = true;

    async fn new(
        source: ScanSource,
        options: &Self::ReadOptions,
        cloud_options: Option<&CloudOptions>,
        row_index: Option<PlSmallStr>,
    ) -> PolarsResult<Self> {
        let has_row_index = row_index.is_some();

        let file_options = FileScanOptions {
            row_index: row_index.map(|name| RowIndex { name, offset: 0 }),
            ..Default::default()
        };

        let mut csv_options = options.clone();
        let mut file_info = isolated_csv_file_info(
            source.as_scan_source_ref(),
            &file_options,
            &mut csv_options,
            cloud_options,
        )?;
        if has_row_index {
            // @HACK: This is really hacky because the CSV schema wrongfully adds the row index.
            let mut schema = file_info.schema.as_ref().clone();
            _ = schema.shift_remove_index(0);
            file_info.schema = Arc::new(schema);
        }
        Ok(Self::new(source, file_info, file_options, csv_options))
    }

    fn with_projection(&mut self, projection: Option<&Bitmap>) {
        self.file_options.with_columns = projection.map(|p| {
            p.true_idx_iter()
                .map(|idx| self.file_info.schema.get_at_index(idx).unwrap().0.clone())
                .collect()
        });
    }
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestrication>) {
        self.file_options.slice = None;
        match row_restriction {
            None => {},
            Some(RowRestrication::Slice(rng)) => {
                self.file_options.slice = Some((rng.start as i64, rng.end - rng.start))
            },
            Some(RowRestrication::Predicate(_)) => unreachable!(),
        }
    }

    async fn unrestricted_row_count(&mut self) -> PolarsResult<IdxSize> {
        let run_async = self.scan_source.run_async();
        let parse_options = self.options.get_parse_options();
        let source = self
            .scan_source
            .as_scan_source_ref()
            .to_memslice_async_assume_latest(run_async)?;

        let mem_slice = {
            let mut out = vec![];
            maybe_decompress_bytes(&source, &mut out)?;

            if out.is_empty() {
                source
            } else {
                MemSlice::from_vec(out)
            }
        };

        let num_rows = polars_io::csv::read::count_rows_from_slice(
            &mem_slice[..],
            parse_options.separator,
            parse_options.quote_char,
            parse_options.comment_prefix.as_ref(),
            parse_options.eol_char,
            self.options.has_header,
        )?;
        let num_rows = IdxSize::try_from(num_rows)
            .map_err(|_| polars_err!(bigidx, ctx = "csv file", size = num_rows))?;
        Ok(num_rows)
    }
    async fn physical_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self.file_info.schema.clone())
    }
}
