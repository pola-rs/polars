use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::io::ipc::read::{Dictionaries, read_dictionary_block};
use async_trait::async_trait;
use polars_async::executor::{self, JoinHandle, TaskPriority};
use polars_async::primitives::wait_group::{WaitGroup, WaitToken};
use polars_buffer::Buffer;
use polars_config::config;
use polars_core::prelude::DataType;
use polars_core::runtime::ASYNC;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, ProjectionInfo, prepare_projection, read_file_metadata,
};
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{ErrString, PolarsError, PolarsResult, polars_err, to_compute_err};
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::ipc::pl_ipc_metadata::{POLARS_IPC_METADATA_KEY, PlIpcMetadata};
use polars_io::utils::byte_source::{
    BufferByteSource, ByteSource, DynByteSource, DynByteSourceBuilder,
};
use polars_io::utils::slice::SplitSlicePosition;
use polars_plan::dsl::ScanSource;
use polars_utils::IdxSize;
use polars_utils::bool::UnsafeBool;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::slice_enum::Slice;
use record_batch_data_fetch::RecordBatchDataFetcher;
use record_batch_decode::RecordBatchDecoder;

use super::multi_scan::reader_interface::BeginReadArgs;
use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use crate::metrics::OptIOMetrics;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::ipc::metadata::read_ipc_metadata_bytes;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::multi_scan::reader_interface::{
    FileReader, FileReaderCallbacks, Projection, calc_row_position_after_slice,
};
use crate::nodes::io_sources::parquet::init::split_to_morsels;
use crate::utils::tokio_handle_ext::AbortOnDropHandle;

pub mod builder;
mod metadata;
mod record_batch_data_fetch;
mod record_batch_decode;

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (pip install polars[rt64])",
));

struct IpcFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    config: Arc<IpcScanOptions>,
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    record_batch_prefetch_sync: RecordBatchPrefetchSync,
    io_metrics: OptIOMetrics,
    verbose: bool,
    init_data: Option<InitializedState>,
    checked: UnsafeBool,
}

struct RecordBatchPrefetchSync {
    prefetch_limit: usize,
    prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,

    /// Waits for the previous reader to finish spawning prefetches.
    prev_all_spawned: Option<WaitGroup>,
    /// Dropped once the current reader has finished spawning prefetches.
    current_all_spawned: Option<WaitToken>,
}

#[derive(Clone)]
struct InitializedState {
    file_metadata: Arc<FileMetadata>,
    file_pl_metadata: Option<Arc<PlIpcMetadata>>,
    byte_source: Arc<DynByteSource>,
    dictionaries: Arc<Option<Dictionaries>>,
}

#[async_trait]
impl FileReader for IpcFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        if self.init_data.is_some() {
            return Ok(());
        }

        let verbose = self.verbose;
        let scan_source = self.scan_source.clone();
        let byte_source_builder = self.byte_source_builder.clone();
        let cloud_options = self.cloud_options.clone();
        let io_metrics = self.io_metrics.clone();

        let byte_source = ASYNC
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

        let mut byte_source = Arc::new(byte_source);

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            let (metadata_bytes, opt_full_bytes) = {
                let byte_source = byte_source.clone();

                ASYNC
                    .spawn(async move { read_ipc_metadata_bytes(&byte_source, verbose).await })
                    .await
                    .unwrap()?
            };

            if let Some(full_bytes) = opt_full_bytes {
                byte_source = Arc::new(DynByteSource::Buffer(BufferByteSource(full_bytes)));
            }

            Arc::new(read_file_metadata(&mut std::io::Cursor::new(
                metadata_bytes,
            ))?)
        };

        let dictionaries = {
            let byte_source_async = byte_source.clone();
            let metadata_async = file_metadata.clone();
            let checked = self.checked;
            let dictionaries = ASYNC
                .spawn(async move {
                    read_dictionaries(&byte_source_async, metadata_async, verbose, checked).await
                })
                .await
                .unwrap()?;
            Arc::new(Some(dictionaries))
        };

        let file_pl_metadata = file_metadata
            .custom_metadata
            .as_ref()
            .and_then(|md| md.get(POLARS_IPC_METADATA_KEY))
            .map(|md_str| serde_json::from_str::<PlIpcMetadata>(md_str))
            .transpose()
            .map_err(to_compute_err)?
            .map(Arc::new);

        self.init_data = Some(InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            file_pl_metadata,
        });

        Ok(())
    }

    fn prepare_read(&mut self) -> PolarsResult<()> {
        let wait_group_this_reader = WaitGroup::default();
        let prefetch_all_spawned_token = wait_group_this_reader.token();

        let prev_wait_group: Option<WaitGroup> = self
            .record_batch_prefetch_sync
            .shared_prefetch_wait_group_slot
            .try_lock()
            .unwrap()
            .replace(wait_group_this_reader);

        self.record_batch_prefetch_sync.prev_all_spawned = prev_wait_group;
        self.record_batch_prefetch_sync.current_all_spawned = Some(prefetch_all_spawned_token);

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        // Initialize.
        let InitializedState {
            file_metadata,
            file_pl_metadata,
            byte_source,
            dictionaries,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            row_index,
            pre_slice: pre_slice_arg,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines,
            disable_morsel_split,
            last_morsel_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    mut n_rows_in_file_tx,
                    mut row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        debug_assert!(!matches!(pre_slice_arg, Some(Slice::Negative { .. })));

        let file_schema_pl = std::cell::LazyCell::new(|| {
            Arc::new(Schema::from_arrow_schema(file_metadata.schema.as_ref()))
        });

        // Handle callbacks that are ready now.
        if let Some(file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.send(file_schema_pl.clone());
        }

        // Always create a slice. If no slice was given, just make the biggest slice possible.
        let slice_range: Range<usize> = pre_slice_arg
            .clone()
            .map_or(0..usize::MAX, Range::<usize>::from);

        // Avoid materializing projection info if we are projecting all the columns of this file.
        let projection_indices: Option<Arc<[usize]>> = if let Some(first_mismatch_idx) =
            (0..usize::min(file_metadata.schema.len(), projected_schema.len())).find(|&i| {
                file_metadata.schema.get_at_index(i).unwrap().0
                    != projected_schema.get_at_index(i).unwrap().0
            }) {
            Some(
                (0..first_mismatch_idx)
                    .chain(
                        (first_mismatch_idx..projected_schema.len()).filter_map(|i| {
                            file_metadata
                                .schema
                                .index_of(projected_schema.get_at_index(i).unwrap().0)
                        }),
                    )
                    .collect(),
            )
        } else if file_metadata.schema.len() > projected_schema.len() {
            // Names match up to projected schema len.
            Some((0..projected_schema.len()).collect())
        } else {
            // Name order matches up to `file_metadata.schema.len()`, we are projecting all columns
            // in this file.
            None
        };

        // Unstable.
        let read_statistics_flags = self.config.record_batch_statistics;

        if verbose {
            eprintln!(
                "[IpcFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                read_record_batch_statistics_flags: {}\
                ",
                projection_indices
                    .as_ref()
                    .map_or(file_metadata.schema.len(), |x| x.len()),
                file_metadata.schema.len(),
                pre_slice_arg,
                read_statistics_flags
            )
        }

        let projection_info: Option<ProjectionInfo> = projection_indices
            .as_deref()
            .map(|indices| prepare_projection(&file_metadata.schema, indices.to_vec()));
        let projection_info = Arc::new(projection_info);

        let schema = projection_info.as_ref().as_ref().map_or(
            file_metadata.schema.as_ref(),
            |ProjectionInfo { schema, .. }| schema,
        );
        let pl_schema = Arc::new(
            schema
                .iter()
                .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
                .collect::<Schema>(),
        );

        // Prepare parameters for Prefetch
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        let record_batch_prefetch_size = self
            .record_batch_prefetch_sync
            .prefetch_limit
            .min(file_metadata.blocks.len())
            .max(1);

        let ideal_morsel_size = get_ideal_morsel_size();

        if verbose {
            eprintln!(
                "[IpcFileReader]: num_pipelines: {num_pipelines}, record_batch_prefetch_size: {record_batch_prefetch_size}, ideal_morsel_size: {ideal_morsel_size}"
            );
            eprintln!(
                "[IpcFileReader]: record batch count: {:?}",
                file_metadata.blocks.len()
            );
        }

        let record_batch_decoder = Arc::new(RecordBatchDecoder {
            file_metadata: file_metadata.clone(),
            pl_schema,
            projection_info,
            dictionaries: dictionaries.clone(),
            row_index,
            read_statistics_flags,
            checked: self.checked,
        });

        // Set up channels.
        let (prefetch_send, mut prefetch_recv) =
            tokio::sync::mpsc::channel(record_batch_prefetch_size);
        let (decode_send, mut decode_recv) = tokio::sync::mpsc::channel(num_pipelines);
        let (mut morsel_send, morsel_recv) = FileReaderOutputSend::new_serial();

        let rb_prefetch_semaphore = Arc::clone(&self.record_batch_prefetch_sync.prefetch_semaphore);
        let rb_prefetch_prev_all_spawned =
            Option::take(&mut self.record_batch_prefetch_sync.prev_all_spawned);
        let rb_prefetch_current_all_spawned =
            Option::take(&mut self.record_batch_prefetch_sync.current_all_spawned);

        // Task: Prefetch.
        let byte_source = byte_source.clone();
        let mut base_rb_metadata_fetch_count: u64 = 0;

        let prefetch_task = AbortOnDropHandle(ASYNC.spawn(async move {
            let record_batch_cum_len = if let Some(file_pl_metadata) = file_pl_metadata {
                struct CumLenWrap(Arc<PlIpcMetadata>);

                impl AsRef<[IdxSize]> for CumLenWrap {
                    fn as_ref(&self) -> &[IdxSize] {
                        self.0.record_batch_cum_len.as_slice()
                    }
                }

                Some(Buffer::from_owner(CumLenWrap(file_pl_metadata)))
            } else if pre_slice_arg.is_some()
                || n_rows_in_file_tx.is_some()
                || row_position_on_end_tx.is_some()
            {
                let mut metadata_ranges: Vec<Range<usize>> = file_metadata
                    .blocks
                    .iter()
                    .map(|block| {
                        block.offset as usize
                            ..block.offset as usize + block.meta_data_length as usize
                    })
                    .collect();

                base_rb_metadata_fetch_count += metadata_ranges.len() as u64;

                if config().verbose() {
                    eprintln!(
                        "[IpcFileReader]: Read all record batch metadata (num_batches: {})",
                        metadata_ranges.len()
                    )
                }

                let record_batch_metadata_map = byte_source
                    .get_ranges(metadata_ranges.as_mut_slice())
                    .await?;

                let mut message_scratch = ScratchVec::default();

                Some(
                    file_metadata
                        .blocks
                        .iter()
                        .map(|block| {
                            let fetched_bytes = record_batch_metadata_map
                                .get(&(block.offset as usize))
                                .unwrap();

                            let mut reader =
                                BlockReader::new(Cursor::new(fetched_bytes.as_slice()));
                            reader
                                .record_batch_num_rows(message_scratch.get())?
                                .try_into()
                                .map_err(|_| polars_err!(ComputeError: LENGTH_LIMIT_MSG))
                        })
                        .scan(0, |offset: &mut IdxSize, v: PolarsResult<IdxSize>| {
                            Some(v.and_then(|num_rows_this_batch| {
                                *offset = offset
                                    .checked_add(num_rows_this_batch)
                                    .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;

                                Ok(*offset)
                            }))
                        })
                        .collect::<PolarsResult<_>>()?,
                )
            } else {
                None
            };

            if let Some(record_batch_cum_len) = record_batch_cum_len.as_deref() {
                let n_rows_in_file = record_batch_cum_len.last().copied().unwrap_or(0);

                if let Some(n_rows_in_file_tx) = n_rows_in_file_tx.take() {
                    _ = n_rows_in_file_tx.send(n_rows_in_file);
                }

                if let Some(row_position_on_end_tx) = row_position_on_end_tx.take() {
                    _ = row_position_on_end_tx.send(calc_row_position_after_slice(
                        n_rows_in_file,
                        pre_slice_arg.clone(),
                    ));
                }
            }

            let record_batch_data_fetcher = RecordBatchDataFetcher {
                file_metadata,
                record_batch_cum_len,

                byte_source,
                memory_prefetch_func,

                subset_projection_idxs: projection_indices,
                pre_slice: pre_slice_arg,

                prefetch_send,
                base_rb_metadata_fetch_count,

                rb_prefetch_semaphore,
                rb_prefetch_current_all_spawned,
            };

            if let Some(rb_prefetch_prev_all_spawned) = rb_prefetch_prev_all_spawned {
                rb_prefetch_prev_all_spawned.wait().await;
            }

            record_batch_data_fetcher.run().await
        }));

        // Receives fetched record batches and synchronizes row position, then calls decode.
        let decode_dispatch_task = AbortOnDropHandle(ASYNC.spawn(async move {
            let mut current_row_offset: IdxSize = 0;

            while let Some((prefetch_task, permit)) = prefetch_recv.recv().await {
                let mut record_batch_data = prefetch_task.await.unwrap()?;

                match record_batch_data.row_offset {
                    Some(row_offset) => current_row_offset = row_offset,
                    None => record_batch_data.row_offset = Some(current_row_offset),
                };

                // Fetch every record batch so we can track the total row count.
                let rb_num_rows = record_batch_data.num_rows;
                let rb_num_rows =
                    IdxSize::try_from(rb_num_rows).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;

                // Only pass to decoder if we need the data.
                let record_batch_position = SplitSlicePosition::split_slice_at_file(
                    current_row_offset as usize,
                    rb_num_rows as usize,
                    slice_range.clone(),
                );

                current_row_offset = current_row_offset
                    .checked_add(rb_num_rows)
                    .ok_or(ROW_COUNT_OVERFLOW_ERR)?;

                match record_batch_position {
                    SplitSlicePosition::Before => continue,
                    SplitSlicePosition::Overlapping(rows_offset, rows_len) => {
                        let record_batch_decoder = record_batch_decoder.clone();
                        let decode_fut = executor::spawn(TaskPriority::High, async move {
                            record_batch_decoder
                                .record_batch_data_to_df(record_batch_data, rows_offset, rows_len)
                                .await
                        });
                        if decode_send.send((decode_fut, permit)).await.is_err() {
                            break;
                        }
                    },
                    SplitSlicePosition::After => break,
                };
            }

            PolarsResult::Ok(())
        }));

        // Task: Distributor.
        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        //
        // `last_morsel_pipelines` is precomputed at the multi-scan layer so the split budget is
        // shared across files in the scan.
        let distribute_task = executor::spawn(TaskPriority::High, async move {
            let mut morsel_seq = MorselSeq::default();
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            // Decode first non-empty morsel.
            let mut next = None;
            loop {
                let Some((decode_fut, permit)) = decode_recv.recv().await else {
                    break;
                };
                let df = decode_fut.await?;
                if df.height() == 0 {
                    continue;
                }

                if disable_morsel_split {
                    if morsel_send
                        .send_morsel(Morsel::new(df, morsel_seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    drop(permit);
                    morsel_seq = morsel_seq.successor();
                    continue;
                }

                next = Some((df, permit));
                break;
            }

            while let Some((df, permit)) = next.take() {
                // Try to decode the next non-empty morsel first, so we know
                // whether the df is the last morsel.

                // Important: Drop this before awaiting the next one, or could
                // deadlock if the permit limit is 1.
                drop(permit);
                loop {
                    let Some((decode_fut, permit)) = decode_recv.recv().await else {
                        break;
                    };
                    let next_df = decode_fut.await?;
                    if next_df.height() == 0 {
                        continue;
                    }
                    next = Some((next_df, permit));
                    break;
                }

                for df in split_to_morsels(
                    &df,
                    ideal_morsel_size,
                    next.is_none(),
                    last_morsel_pipelines,
                ) {
                    if morsel_send
                        .send_morsel(Morsel::new(df, morsel_seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    morsel_seq = morsel_seq.successor();
                }
            }
            PolarsResult::Ok(())
        });

        // Orchestration.
        let join_task = ASYNC.spawn(async move {
            prefetch_task.await.unwrap()?;
            decode_dispatch_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        let handle = AbortOnDropHandle(join_task);

        Ok((
            morsel_recv,
            executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }
}

async fn read_dictionaries(
    byte_source: &DynByteSource,
    file_metadata: Arc<FileMetadata>,
    verbose: bool,
    checked: UnsafeBool,
) -> PolarsResult<Dictionaries> {
    let blocks = if let Some(blocks) = &file_metadata.dictionaries {
        blocks
    } else {
        return Ok(Dictionaries::default());
    };

    if verbose {
        eprintln!("[IpcFileReader]: reading dictionaries ({:?})", blocks.len());
    }

    let mut dictionaries = Dictionaries::default();

    let mut message_scratch = Vec::new();
    let mut dictionary_scratch = Vec::new();

    for block in blocks {
        let range = block.offset as usize
            ..block.offset as usize + block.meta_data_length as usize + block.body_length as usize;
        let bytes = byte_source.get_range(range).await?;

        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));

        read_dictionary_block(
            &mut reader.reader,
            file_metadata.as_ref(),
            block,
            true,
            &mut dictionaries,
            &mut message_scratch,
            &mut dictionary_scratch,
            checked,
        )?;
    }

    Ok(dictionaries)
}
