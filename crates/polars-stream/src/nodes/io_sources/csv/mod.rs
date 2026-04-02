pub mod builder;
mod chunk_reader;
mod line_batch_source;

use std::iter::Iterator;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use chunk_reader::ChunkReader;
use line_batch_source::{LineBatch, LineBatchSource};
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::metrics::OptIOMetrics;
use polars_io::pl_async;
use polars_io::prelude::_csv_read_internal::CountLines;
use polars_io::prelude::CsvReadOptions;
use polars_io::prelude::streaming::read_until_start_and_infer_schema;
use polars_io::utils::byte_source::{ByteSource, DynByteSource, DynByteSourceBuilder};
use polars_io::utils::compression::{ByteSourceReader, SupportedCompression};
use polars_io::utils::stream_buf_reader::{ReaderSource, StreamBufReader};
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::slice_enum::Slice;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, FileReader, FileReaderCallbacks};
use super::shared::chunk_data_fetch::ChunkDataFetcher;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::SourceToken;
use crate::nodes::TaskPriority;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::utils::tokio_handle_ext;

/// Read all rows in the chunk
const NO_SLICE: (usize, usize) = (0, usize::MAX);
/// This is used if we finish the slice but still need a row count. It signals to the workers to
/// go into line-counting mode where they can skip parsing the chunks.
const SLICE_ENDED: (usize, usize) = (usize::MAX, 0);

struct CsvFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    options: Arc<CsvReadOptions>,
    verbose: bool,
    pub byte_source_builder: DynByteSourceBuilder,
    pub chunk_prefetch_sync: ChunkPrefetchSync,
    pub init_data: Option<InitializedState>,
    pub io_metrics: OptIOMetrics,
}

pub(crate) struct ChunkPrefetchSync {
    pub(crate) prefetch_limit: usize,
    pub(crate) prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    pub(crate) shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,

    /// Waits for the previous reader to finish spawning prefetches.
    pub(crate) prev_all_spawned: Option<WaitGroup>,
    /// Dropped once the current reader has finished spawning prefetches.
    pub(crate) current_all_spawned: Option<WaitToken>,
}

#[derive(Clone)]
pub struct InitializedState {
    file_size: usize,
    compression: Option<SupportedCompression>,
    byte_source: Arc<DynByteSource>,
}

#[async_trait]
impl FileReader for CsvFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        if self.init_data.is_some() {
            return Ok(());
        }

        let scan_source = self.scan_source.clone();
        let byte_source_builder = self.byte_source_builder.clone();
        let cloud_options = self.cloud_options.clone();
        let io_metrics = self.io_metrics.clone();

        let byte_source = pl_async::get_runtime()
            .spawn(async move {
                scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(
                        &byte_source_builder,
                        cloud_options.as_deref(),
                        io_metrics.0,
                    )
                    .await
            })
            .await
            .unwrap()?;
        let byte_source = Arc::new(byte_source);

        // @TODO: Refactor FileInfo so we can re-use the file_size value from the planning stage.
        let file_size = {
            let byte_source = byte_source.clone();
            pl_async::get_runtime()
                .spawn(async move { byte_source.get_size().await })
                .await
                .unwrap()?
        };

        let compression = if file_size >= 4 {
            let byte_source = byte_source.clone();
            let magic_range = 0..4;
            let magic_bytes = pl_async::get_runtime()
                .spawn(async move { byte_source.get_range(magic_range).await })
                .await
                .unwrap()?;
            SupportedCompression::check(&magic_bytes)
        } else {
            None
        };

        self.init_data = Some(InitializedState {
            file_size,
            compression,
            byte_source,
        });

        Ok(())
    }

    fn prepare_read(&mut self) -> PolarsResult<()> {
        let wait_group_this_reader = WaitGroup::default();
        let prefetch_all_spawned_token = wait_group_this_reader.token();

        let prev_wait_group: Option<WaitGroup> = self
            .chunk_prefetch_sync
            .shared_prefetch_wait_group_slot
            .try_lock()
            .unwrap()
            .replace(wait_group_this_reader);

        self.chunk_prefetch_sync.prev_all_spawned = prev_wait_group;
        self.chunk_prefetch_sync.current_all_spawned = Some(prefetch_all_spawned_token);

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        // Initialize.
        let InitializedState {
            file_size,
            compression,
            byte_source,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            // Because we currently only support PRE_SLICE we don't need to handle row index here.
            row_index,
            pre_slice,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines,
            disable_morsel_split: _,
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

        // There are two byte sourcing strategies `ReaderSource`: (a) async parallel prefetch using a
        // streaming pipeline, or (b) memory-mapped, only to be used for uncompressed local files.
        // The `compressed_reader` (of type `ByteSourceReader`) abstracts these source types.
        // The `use_async_prefetch` flag controls the optional pipeline startup behavior.
        let use_async_prefetch =
            !(matches!(byte_source.as_ref(), &DynByteSource::Buffer(_)) && compression.is_none());

        const ASSUMED_COMPRESSION_RATIO: usize = 4;
        let decompressed_file_size_hint = match compression {
            None => Some(file_size),
            Some(_) => Some(file_size * ASSUMED_COMPRESSION_RATIO),
        };

        // Unify the two source options (uncompressed local file mmapp'ed, or streaming async with transparent
        // decompression), into one unified reader object.
        let mut reader: ByteSourceReader<ReaderSource> = if use_async_prefetch {
            // Prepare parameters for Prefetch task.
            const DEFAULT_CSV_CHUNK_SIZE: usize = 32 * 1024 * 1024;
            let memory_prefetch_func = get_memory_prefetch_func(verbose);
            let chunk_size = std::env::var("POLARS_CSV_CHUNK_SIZE")
                .map(|x| {
                    x.parse::<NonZeroUsize>()
                        .unwrap_or_else(|_| panic!("invalid value for POLARS_CSV_CHUNK_SIZE: {x}"))
                        .get()
                })
                .unwrap_or(DEFAULT_CSV_CHUNK_SIZE);

            let prefetch_limit = self
                .chunk_prefetch_sync
                .prefetch_limit
                .min(file_size.div_ceil(chunk_size))
                .max(1);

            let (prefetch_send, prefetch_recv) = tokio::sync::mpsc::channel(prefetch_limit);

            // Task: Prefetch.
            // Initiate parallel downloads of raw data chunks.
            let byte_source = byte_source.clone();
            let prefetch_task = {
                let io_runtime = polars_io::pl_async::get_runtime();

                let prefetch_semaphore = Arc::clone(&self.chunk_prefetch_sync.prefetch_semaphore);
                let prefetch_prev_all_spawned =
                    Option::take(&mut self.chunk_prefetch_sync.prev_all_spawned);
                let prefetch_current_all_spawned =
                    Option::take(&mut self.chunk_prefetch_sync.current_all_spawned);

                tokio_handle_ext::AbortOnDropHandle(io_runtime.spawn(async move {
                    let mut chunk_data_fetcher = ChunkDataFetcher {
                        memory_prefetch_func,
                        byte_source,
                        file_size,
                        chunk_size,
                        prefetch_send,
                        prefetch_semaphore,
                        prefetch_current_all_spawned,
                    };

                    if let Some(prefetch_prev_all_spawned) = prefetch_prev_all_spawned {
                        prefetch_prev_all_spawned.wait().await;
                    }

                    chunk_data_fetcher.run().await?;

                    Ok(())
                }))
            };

            // Wrap into ByteSourceReader to enable sync `BufRead` access.
            let stream_buf_reader = StreamBufReader::new(prefetch_recv, prefetch_task);
            ByteSourceReader::try_new(ReaderSource::Streaming(stream_buf_reader), compression)?
        } else {
            let memslice = self
                .scan_source
                .as_scan_source_ref()
                .to_buffer_async_assume_latest(self.scan_source.run_async())?;

            ByteSourceReader::from_memory(memslice)?
        };

        let (inferred_schema, base_leftover) = read_until_start_and_infer_schema(
            &self.options,
            Some(projected_schema.clone()),
            decompressed_file_size_hint,
            None,
            &mut reader,
        )?;

        let used_schema = Arc::new(inferred_schema);

        if let Some(tx) = file_schema_tx {
            _ = tx.send(used_schema.clone())
        }

        let projection: Vec<usize> = projected_schema
            .iter_names()
            .filter_map(|name| used_schema.index_of(name))
            .collect();

        if verbose {
            eprintln!(
                "[CsvFileReader]: project: {} / {}, \
                slice: {:?}, \
                use_async_prefetch: {}",
                projection.len(),
                used_schema.len(),
                &pre_slice,
                use_async_prefetch
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
                base_leftover,
                reader,
                line_counter,
                line_batch_tx,
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
