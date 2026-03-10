pub mod builder;

use std::cmp::Reverse;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use line_batch_processor::{LineBatchProcessor, LineBatchProcessorOutputPort};
use negative_slice_pass::MorselStreamReverser;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::metrics::OptIOMetrics;
use polars_io::pl_async;
use polars_io::utils::byte_source::{ByteSource, DynByteSource, DynByteSourceBuilder};
use polars_io::utils::compression::{ByteSourceReader, SupportedCompression};
use polars_io::utils::stream_buf_reader::{ReaderSource, StreamBufReader};
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::priority::Priority;
use polars_utils::slice_enum::Slice;
use row_index_limit_pass::ApplyRowIndexOrLimit;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, FileReader, FileReaderCallbacks};
use super::shared::chunk_data_fetch::ChunkDataFetcher;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::oneshot_channel;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::ndjson::chunk_reader::ChunkReaderBuilder;
use crate::nodes::io_sources::ndjson::line_batch_distributor::RowSkipper;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::tokio_handle_ext;
pub(super) mod chunk_reader;
mod line_batch_distributor;
mod line_batch_processor;
mod negative_slice_pass;
mod row_index_limit_pass;

pub struct NDJsonFileReader {
    pub scan_source: ScanSource,
    pub cloud_options: Option<Arc<CloudOptions>>,
    pub chunk_reader_builder: ChunkReaderBuilder,
    pub count_rows_fn: fn(&[u8]) -> usize,
    pub verbose: bool,
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
impl FileReader for NDJsonFileReader {
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
            mut row_index,
            pre_slice,

            num_pipelines,
            disable_morsel_split: _,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },

            predicate: None,
            cast_columns_policy: _,
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        let is_empty_slice = pre_slice.as_ref().is_some_and(|x| x.len() == 0);
        let is_negative_slice = matches!(pre_slice, Some(Slice::Negative { .. }));

        // There are two byte sourcing strategies `ReaderSource`: (a) async parallel prefetch using a
        // streaming pipeline, or (b) memory-mapped, only to be used for uncompressed local files.
        // The `compressed_reader` (of type `ByteSourceReader`) abstracts these source types.
        // The `use_async_prefetch` flag controls the optional pipeline startup behavior.
        let use_async_prefetch =
            !(matches!(byte_source.as_ref(), &DynByteSource::Buffer(_)) && compression.is_none());

        // NDJSON: We just use the projected schema - the parser will automatically append NULL if
        // the field is not found.
        //
        // TODO
        // We currently always use the projected dtype, but this may cause
        // issues e.g. with temporal types. This can be improved to better choose
        // between the 2 dtypes.
        let schema = projected_schema;

        if let Some(tx) = file_schema_tx {
            _ = tx.send(schema.clone())
        }

        // Convert (offset, len) to Range
        // Note: This is converted to right-to-left for negative slice (i.e. range.start is position
        // from end).
        let global_slice: Option<Range<usize>> = if let Some(slice) = pre_slice.clone() {
            match slice {
                Slice::Positive { offset, len } => Some(offset..offset.saturating_add(len)),
                Slice::Negative {
                    offset_from_end,
                    len,
                } => {
                    // array: [_ _ _ _ _]
                    // slice: [    _ _  ]
                    // in:    offset_from_end: 3, len: 2
                    // out:   1..3 (right-to-left)
                    Some(offset_from_end.saturating_sub(len)..offset_from_end)
                },
            }
        } else {
            None
        };

        let (total_row_count_tx, total_row_count_rx) = if is_negative_slice && row_index.is_some() {
            let (tx, rx) = oneshot_channel::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let needs_total_row_count = total_row_count_tx.is_some()
            || n_rows_in_file_tx.is_some()
            || (row_position_on_end_tx.is_some()
                && matches!(pre_slice, Some(Slice::Negative { .. })));

        if verbose {
            eprintln!(
                "[NDJsonFileReader]: \
                project: {}, \
                global_slice: {:?}, \
                row_index: {:?}, \
                is_negative_slice: {}, \
                use_async_prefetch: {}",
                schema.len(),
                &global_slice,
                &row_index,
                is_negative_slice,
                use_async_prefetch
            );
        }

        // Note: This counts from the end of file for negative slice.
        let n_rows_to_skip = global_slice.as_ref().map_or(0, |x| x.start);

        let (opt_linearizer, mut linearizer_inserters) =
            if global_slice.is_some() || row_index.is_some() {
                let (a, b) =
                    Linearizer::<Priority<Reverse<MorselSeq>, DataFrame>>::new(num_pipelines, 1);
                (Some(a), b)
            } else {
                (None, vec![])
            };

        let output_to_linearizer = opt_linearizer.is_some();
        let mut output_port = None;

        let opt_post_process_handle = if is_negative_slice {
            // Note: This is right-to-left
            let negative_slice = global_slice.unwrap();

            if verbose {
                eprintln!("[NDJsonFileReader]: Initialize morsel stream reverser");
            }

            let (morsel_senders, rx) = FileReaderOutputSend::new_parallel(num_pipelines);
            output_port = Some(rx);

            Some(AbortOnDropHandle::new(spawn(
                TaskPriority::High,
                MorselStreamReverser {
                    morsel_receiver: opt_linearizer.unwrap(),
                    morsel_senders,
                    offset_len_rtl: (
                        negative_slice.start,
                        negative_slice.end - negative_slice.start,
                    ),
                    // The correct row index offset can only be known after total row count is
                    // available. This is handled by the MorselStreamReverser.
                    row_index: row_index.take().map(|x| (x, total_row_count_rx.unwrap())),
                    verbose,
                }
                .run(),
            )))
        } else if global_slice.is_some() || row_index.is_some() {
            let mut row_index = row_index.take();

            if verbose {
                eprintln!("[NDJsonFileReader]: Initialize ApplyRowIndexOrLimit");
            }

            if let Some(ri) = row_index.as_mut() {
                // Update the row index offset according to the slice start.
                let Some(v) = ri.offset.checked_add(n_rows_to_skip as IdxSize) else {
                    let offset = ri.offset;

                    polars_bail!(
                        ComputeError:
                        "row_index with offset {} overflows at {} rows",
                        offset, n_rows_to_skip
                    )
                };
                ri.offset = v;
            }

            let (morsel_tx, rx) = FileReaderOutputSend::new_serial();
            output_port = Some(rx);

            let limit = global_slice.as_ref().map(|x| x.len());

            let task = ApplyRowIndexOrLimit {
                morsel_receiver: opt_linearizer.unwrap(),
                morsel_tx,
                // Note: The line batch distributor handles skipping lines until the offset,
                // we only need to handle the limit here.
                limit,
                row_index,
                verbose,
            };

            if is_empty_slice {
                None
            } else {
                Some(AbortOnDropHandle::new(spawn(
                    TaskPriority::High,
                    task.run(),
                )))
            }
        } else {
            None
        };

        let chunk_reader = self.chunk_reader_builder.build(schema);

        let (line_batch_distribute_tx, line_batch_distribute_receivers) =
            distributor_channel(num_pipelines, 1);

        let mut morsel_senders = if !output_to_linearizer {
            let (senders, outp) = FileReaderOutputSend::new_parallel(num_pipelines);
            assert!(output_port.is_none());
            output_port = Some(outp);
            senders
        } else {
            vec![]
        };

        // Initialize in reverse as we want to manually pop from either the linearizer or the phase receivers depending
        // on if we have negative slice.
        let line_batch_processor_handles = line_batch_distribute_receivers
            .into_iter()
            .enumerate()
            .rev()
            .map(|(worker_idx, line_batch_rx)| {
                let chunk_reader = chunk_reader.clone();
                let count_rows_fn = self.count_rows_fn;
                // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
                let source_token = SourceToken::new();

                AbortOnDropHandle::new(spawn(
                    TaskPriority::Low,
                    LineBatchProcessor {
                        worker_idx,

                        chunk_reader,
                        count_rows_fn,

                        line_batch_rx,
                        output_port: if is_empty_slice {
                            LineBatchProcessorOutputPort::Closed
                        } else if output_to_linearizer {
                            LineBatchProcessorOutputPort::Linearize {
                                tx: linearizer_inserters.pop().unwrap(),
                            }
                        } else {
                            LineBatchProcessorOutputPort::Direct {
                                tx: morsel_senders.pop().unwrap(),
                                source_token,
                            }
                        },
                        needs_total_row_count,

                        // Only log from the last worker to prevent flooding output.
                        verbose: verbose && worker_idx == num_pipelines - 1,
                    }
                    .run(),
                ))
            })
            .collect::<Vec<_>>();

        let row_skipper = RowSkipper {
            cfg_n_rows_to_skip: n_rows_to_skip,
            n_rows_skipped: 0,
            is_line: self.chunk_reader_builder.is_line_fn(),
            reverse: is_negative_slice,
        };

        // Unify the two source options (uncompressed local file mmapp'ed, or streaming async with transparent
        // decompression), into one unified reader object.
        let byte_source_reader: ByteSourceReader<ReaderSource> = if use_async_prefetch {
            // Prepare parameters for Prefetch task.
            const DEFAULT_NDJSON_CHUNK_SIZE: usize = 32 * 1024 * 1024;
            let memory_prefetch_func = get_memory_prefetch_func(verbose);
            let chunk_size = std::env::var("POLARS_NDJSON_CHUNK_SIZE")
                .map(|x| {
                    x.parse::<NonZeroUsize>()
                        .unwrap_or_else(|_| {
                            panic!("invalid value for POLARS_NDJSON_CHUNK_SIZE: {x}")
                        })
                        .get()
                })
                .unwrap_or(DEFAULT_NDJSON_CHUNK_SIZE);

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

        const ASSUMED_COMPRESSION_RATIO: usize = 4;
        let uncompressed_file_size_hint = Some(match compression {
            Some(_) => file_size * ASSUMED_COMPRESSION_RATIO,
            None => file_size,
        });

        let line_batch_distributor_task_handle = AbortOnDropHandle::new(spawn(
            TaskPriority::Low,
            line_batch_distributor::LineBatchDistributor {
                reader: byte_source_reader,
                reverse: is_negative_slice,
                row_skipper,
                line_batch_distribute_tx,
                uncompressed_file_size_hint,
            }
            .run(),
        ));

        // Task. Finishing handle.
        let finishing_handle = spawn(TaskPriority::Low, async move {
            // Number of rows skipped by the line batch distributor.
            let n_rows_skipped: usize = line_batch_distributor_task_handle.await?;

            // Number of rows processed by the line batch processors.
            let mut n_rows_processed: usize = 0;

            if verbose {
                eprintln!("[NDJsonFileReader]: line batch distributor handle returned");
            }

            for handle in line_batch_processor_handles {
                n_rows_processed = n_rows_processed.saturating_add(handle.await?);
            }

            let total_row_count =
                needs_total_row_count.then_some(n_rows_skipped.saturating_add(n_rows_processed));

            if verbose {
                eprintln!("[NDJsonFileReader]: line batch processor handles returned");
            }

            if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                let n = match pre_slice {
                    None => n_rows_skipped.saturating_add(n_rows_processed),

                    Some(Slice::Positive { offset, len }) => n_rows_skipped
                        .saturating_add(n_rows_processed)
                        .min(offset.saturating_add(len)),

                    Some(Slice::Negative { .. }) => {
                        total_row_count.unwrap().saturating_sub(n_rows_skipped)
                    },
                };

                let n = IdxSize::try_from(n)
                    .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = n))?;

                _ = row_position_on_end_tx.send(n);
            }

            if let Some(tx) = total_row_count_tx {
                let total_row_count = total_row_count.unwrap();

                if verbose {
                    eprintln!(
                        "[NDJsonFileReader]: \
                        send total row count: {total_row_count}"
                    )
                }
                _ = tx.send(total_row_count);
            }

            if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                let total_row_count = total_row_count.unwrap();

                if verbose {
                    eprintln!("[NDJsonFileReader]: send n_rows_in_file: {total_row_count}");
                }

                let num_rows = total_row_count;
                let num_rows = IdxSize::try_from(num_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "ndjson file", size = num_rows))?;
                _ = n_rows_in_file_tx.send(num_rows);
            }

            if let Some(handle) = opt_post_process_handle {
                handle.await?;
            }

            if verbose {
                eprintln!("[NDJsonFileReader]: returning");
            }

            Ok(())
        });

        Ok((output_port.unwrap(), finishing_handle))
    }
}
