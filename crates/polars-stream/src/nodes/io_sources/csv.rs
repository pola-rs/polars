use std::future::Future;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_core::prelude::{AnyValue, DataType, Field};
use polars_core::scalar::Scalar;
use polars_core::schema::{SchemaExt, SchemaRef};
#[cfg(feature = "dtype-categorical")]
use polars_core::StringCacheHolder;
use polars_error::{polars_bail, PolarsResult};
use polars_io::prelude::_csv_read_internal::{
    cast_columns, find_starting_point, prepare_csv_schema, read_chunk, CountLines,
    NullValuesCompiled,
};
use polars_io::prelude::buffer::validate_utf8;
use polars_io::prelude::{CsvEncoding, CsvParseOptions, CsvReadOptions};
use polars_io::utils::compression::maybe_decompress_bytes;
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::RowIndex;
use polars_plan::plans::{FileInfo, ScanSources};
use polars_plan::prelude::FileScanOptions;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use crate::async_executor;
use crate::async_primitives::connector::connector;
use crate::async_primitives::wait_group::{IndexedWaitGroup, WaitToken};
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::{MorselSeq, TaskPriority};

struct LineBatch {
    bytes: MemSlice,
    n_lines: usize,
    slice: (usize, usize),
    row_offset: usize,
    morsel_seq: MorselSeq,
    wait_token: WaitToken,
    path_name: Option<PlSmallStr>,
}

type AsyncTaskData = (
    Vec<crate::async_primitives::connector::Receiver<LineBatch>>,
    Arc<ChunkReader>,
    async_executor::AbortOnDropHandle<PolarsResult<()>>,
);

pub struct CsvSourceNode {
    scan_sources: ScanSources,
    file_info: FileInfo,
    file_options: FileScanOptions,
    options: CsvReadOptions,
    schema: Option<SchemaRef>,
    num_pipelines: usize,
    async_task_data: Arc<tokio::sync::Mutex<Option<AsyncTaskData>>>,
    is_finished: Arc<AtomicBool>,
    verbose: bool,
}

impl CsvSourceNode {
    pub fn new(
        scan_sources: ScanSources,
        file_info: FileInfo,
        file_options: FileScanOptions,
        options: CsvReadOptions,
    ) -> Self {
        let verbose = config::verbose();

        Self {
            scan_sources,
            file_info,
            file_options,
            options,
            schema: None,
            num_pipelines: 0,
            async_task_data: Arc::new(tokio::sync::Mutex::new(None)),
            is_finished: Arc::new(AtomicBool::new(false)),
            verbose,
        }
    }
}

impl ComputeNode for CsvSourceNode {
    fn name(&self) -> &str {
        "csv_source"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;

        if self.verbose {
            eprintln!("[CsvSource]: initialize");
        }

        self.schema = Some(self.file_info.reader_schema.take().unwrap().unwrap_right());
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        use std::sync::atomic::Ordering;

        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self.is_finished.load(Ordering::Relaxed) {
            send[0] = PortState::Done;
            assert!(
                self.async_task_data.try_lock().unwrap().is_none(),
                "should have already been shut down"
            );
        } else if send[0] == PortState::Done {
            {
                // Early shutdown - our port state was set to `Done` by the downstream nodes.
                self.shutdown_in_background();
            };
            self.is_finished.store(true, Ordering::Relaxed);
        } else {
            send[0] = PortState::Ready
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        use std::sync::atomic::Ordering;

        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);
        assert!(!self.is_finished.load(Ordering::Relaxed));

        let morsel_senders = send_ports[0].take().unwrap().parallel();

        let mut async_task_data_guard = {
            let guard = self.async_task_data.try_lock().unwrap();

            if guard.is_some() {
                guard
            } else {
                drop(guard);
                let v = self.init_line_batch_source();
                let mut guard = self.async_task_data.try_lock().unwrap();
                guard.replace(v);
                guard
            }
        };

        let (line_batch_receivers, chunk_reader, _) = async_task_data_guard.as_mut().unwrap();

        assert_eq!(line_batch_receivers.len(), morsel_senders.len());

        let is_finished = self.is_finished.clone();
        let source_token = SourceToken::new();

        let task_handles = line_batch_receivers
            .drain(..)
            .zip(morsel_senders)
            .map(|(mut line_batch_rx, mut morsel_tx)| {
                let is_finished = is_finished.clone();
                let chunk_reader = chunk_reader.clone();
                let source_token = source_token.clone();

                scope.spawn_task(TaskPriority::Low, async move {
                    loop {
                        let Ok(LineBatch {
                            bytes,
                            n_lines,
                            slice: (offset, len),
                            row_offset,
                            morsel_seq,
                            wait_token,
                            mut path_name,
                        }) = line_batch_rx.recv().await
                        else {
                            is_finished.store(true, Ordering::Relaxed);
                            break;
                        };

                        let mut df =
                            chunk_reader.read_chunk(&bytes, n_lines, (offset, len), row_offset)?;

                        if let Some(path_name) = path_name.take() {
                            unsafe {
                                df.with_column_unchecked(
                                    Scalar::new(DataType::String, AnyValue::StringOwned(path_name))
                                        .into_column(
                                            chunk_reader.include_file_paths.clone().unwrap(),
                                        )
                                        .new_from_index(0, df.height()),
                                )
                            };
                        }

                        let mut morsel = Morsel::new(df, morsel_seq, source_token.clone());
                        morsel.set_consume_token(wait_token);

                        if morsel_tx.send(morsel).await.is_err() {
                            break;
                        }

                        if source_token.stop_requested() {
                            break;
                        }
                    }

                    PolarsResult::Ok(line_batch_rx)
                })
            })
            .collect::<Vec<_>>();

        drop(async_task_data_guard);

        let async_task_data = self.async_task_data.clone();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            {
                let mut async_task_data_guard = async_task_data.try_lock().unwrap();
                let (line_batch_receivers, ..) = async_task_data_guard.as_mut().unwrap();

                for handle in task_handles {
                    line_batch_receivers.push(handle.await?);
                }
            }

            if self.is_finished.load(Ordering::Relaxed) {
                self.shutdown().await?;
            }

            Ok(())
        }))
    }
}

impl CsvSourceNode {
    fn init_line_batch_source(&mut self) -> AsyncTaskData {
        let verbose = self.verbose;

        let (mut line_batch_senders, line_batch_receivers): (Vec<_>, Vec<_>) =
            (0..self.num_pipelines).map(|_| connector()).unzip();

        let scan_sources = self.scan_sources.clone();
        let run_async = scan_sources.is_cloud_url() || config::force_async();
        let num_pipelines = self.num_pipelines;

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
        let include_file_paths = self.file_options.include_file_paths.is_some();

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

                let mut wait_groups = (0..num_pipelines)
                    .map(|index| IndexedWaitGroup::new(index).wait())
                    .collect::<FuturesUnordered<_>>();
                let morsel_seq_ref = &mut MorselSeq::default();
                let current_row_offset_ref = &mut 0usize;

                let n_parts_hint = num_pipelines * 16;

                let line_counter = CountLines::new(quote_char, eol_char);

                let comment_prefix = comment_prefix.as_ref();

                'main: for (i, v) in scan_sources
                    .iter()
                    .map(|x| {
                        let bytes = x.to_memslice_async_assume_latest(run_async)?;
                        PolarsResult::Ok((
                            bytes,
                            include_file_paths.then(|| x.to_include_path_name().into()),
                        ))
                    })
                    .enumerate()
                {
                    if verbose {
                        eprintln!(
                            "[CsvSource]: Start line splitting for file {} / {}",
                            1 + i,
                            scan_sources.len()
                        );
                    }
                    let (mem_slice, path_name) = v?;
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
                        comment_prefix,
                        has_header,
                    )?;

                    let mut bytes = &bytes[i..];

                    let mut chunk_size = {
                        let max_chunk_size = 16 * 1024 * 1024;
                        let chunk_size = if global_slice.is_some() {
                            max_chunk_size
                        } else {
                            std::cmp::min(bytes.len() / n_parts_hint, max_chunk_size)
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
                                SplitSlicePosition::After => break 'main,
                            }
                        } else {
                            // (0, 0) is interpreted as no slicing
                            (0, 0)
                        };

                        let mut mem_slice_this_chunk =
                            mem_slice.slice(slice_start..slice_start + position);

                        let morsel_seq = *morsel_seq_ref;
                        *morsel_seq_ref = morsel_seq.successor();

                        let Some(mut indexed_wait_group) = wait_groups.next().await else {
                            break;
                        };

                        let mut path_name = path_name.clone();

                        loop {
                            use crate::async_primitives::connector::SendError;

                            let channel_index = indexed_wait_group.index();
                            let wait_token = indexed_wait_group.token();

                            match line_batch_senders[channel_index].try_send(LineBatch {
                                bytes: mem_slice_this_chunk,
                                n_lines: count,
                                slice,
                                row_offset: current_row_offset,
                                morsel_seq,
                                wait_token,
                                path_name,
                            }) {
                                Ok(_) => {
                                    wait_groups.push(indexed_wait_group.wait());
                                    break;
                                },
                                Err(SendError::Closed(v)) => {
                                    mem_slice_this_chunk = v.bytes;
                                    path_name = v.path_name;
                                },
                                Err(SendError::Full(_)) => unreachable!(),
                            }

                            let Some(v) = wait_groups.next().await else {
                                break 'main; // All channels closed
                            };

                            indexed_wait_group = v;
                        }
                    }
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
            self.file_options.include_file_paths.clone(),
        )
    }

    /// # Panics
    /// Panics if called more than once.
    async fn shutdown_impl(
        async_task_data: Arc<tokio::sync::Mutex<Option<AsyncTaskData>>>,
        verbose: bool,
    ) -> PolarsResult<()> {
        if verbose {
            eprintln!("[CsvSource]: Shutting down");
        }

        let (line_batch_receivers, _chunk_reader, task_handle) =
            async_task_data.try_lock().unwrap().take().unwrap();

        drop(line_batch_receivers);
        // Join on the producer handle to catch errors/panics.
        // Safety
        // * We dropped the receivers on the line above
        // * This function is only called once.
        task_handle.await
    }

    fn shutdown(&self) -> impl Future<Output = PolarsResult<()>> {
        if self.verbose {
            eprintln!("[CsvSource]: Shutdown via `shutdown()`");
        }
        Self::shutdown_impl(self.async_task_data.clone(), self.verbose)
    }

    fn shutdown_in_background(&self) {
        if self.verbose {
            eprintln!("[CsvSource]: Shutdown via `shutdown_in_background()`");
        }
        let async_task_data = self.async_task_data.clone();
        polars_io::pl_async::get_runtime()
            .spawn(Self::shutdown_impl(async_task_data, self.verbose));
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
    include_file_paths: Option<PlSmallStr>,
}

impl ChunkReader {
    fn try_new(
        options: &mut CsvReadOptions,
        reader_schema: &SchemaRef,
        with_columns: Option<&[PlSmallStr]>,
        row_index: Option<RowIndex>,
        include_file_paths: Option<PlSmallStr>,
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
            include_file_paths,
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
