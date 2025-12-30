use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::io::ipc::read::{Dictionaries, read_dictionary_block};
use async_trait::async_trait;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, ProjectionInfo, get_row_count_from_blocks, prepare_projection,
    read_file_metadata,
};
use polars_error::{ErrString, PolarsError, PolarsResult, feature_gated, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::pl_async;
#[cfg(feature = "cloud")]
use polars_io::pl_async::get_runtime;
use polars_io::utils::byte_source::{
    ByteSource, DynByteSource, DynByteSourceBuilder, MemSliceByteSource,
};
use polars_plan::dsl::{ScanSource, ScanSourceRef};
use polars_utils::IdxSize;
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::plpath::PlPathRef;
use polars_utils::slice_enum::Slice;
use record_batch_data_fetch::RecordBatchDataFetcher;
use record_batch_decode::RecordBatchDecoder;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, calc_row_position_after_slice};
use crate::async_executor::{self, JoinHandle, TaskPriority, spawn};
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::multi_scan::reader_interface::{
    FileReader, FileReaderCallbacks, Projection,
};
use crate::nodes::io_sources::parquet::init::split_to_morsels;
use crate::utils::tokio_handle_ext::AbortOnDropHandle;

pub mod builder;
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
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    record_batch_prefetch_sync: RecordBatchPrefetchSync,
    verbose: bool,
    init_data: Option<InitializedState>,
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
    byte_source: Arc<DynByteSource>,
    dictionaries: Arc<Option<Dictionaries>>,
    // Lazily initialized - getting this involves iterating record batches.
    n_rows_in_file: Option<IdxSize>,
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

        let byte_source = pl_async::get_runtime()
            .spawn(async move {
                scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(&byte_source_builder, cloud_options.as_deref())
                    .await
            })
            .await
            .unwrap()?;

        let byte_source = Arc::new(byte_source);

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            self.fetch_file_metadata().await?
        };

        let dictionaries = {
            let byte_source_async = byte_source.clone();
            let metadata_async = file_metadata.clone();
            let dictionaries = pl_async::get_runtime()
                .spawn(async move {
                    read_dictionaries(&byte_source_async, metadata_async, verbose).await
                })
                .await
                .unwrap()?;
            Arc::new(Some(dictionaries))
        };

        self.init_data = Some(InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            n_rows_in_file: None,
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
            byte_source,
            dictionaries,
            n_rows_in_file: _,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            row_index,
            pre_slice: pre_slice_arg,
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

        let file_schema_pl = std::cell::LazyCell::new(|| {
            Arc::new(Schema::from_arrow_schema(file_metadata.schema.as_ref()))
        });

        // Handle callbacks that are ready now.
        if let Some(file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.send(file_schema_pl.clone());
        }

        // Normalize slice.
        // @NOTE. When this is expensive (e.g., when downloading from cloud) and not strictly required,
        // we skip this normalization and handle it later. Negative slicing requires normalization.
        let normalized_pre_slice = if let Some(pre_slice) = pre_slice_arg.clone()
            && !(self.scan_source.is_cloud_url() && matches!(pre_slice, Slice::Positive { .. }))
        {
            Some(pre_slice.restrict_to_bounds(usize::try_from(self._n_rows_in_file()?).unwrap()))
        } else {
            pre_slice_arg.clone()
        };

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();
            let n_rows = self._n_rows_in_file()?;

            if verbose {
                eprintln!(
                    "[IpcFileReader]: early return: \
                    n_rows_in_file: {}, \
                    pre_slice: {:?}, \
                    resolved_pre_slice: {:?} \
                    ",
                    n_rows, pre_slice_arg, normalized_pre_slice
                )
            }

            // Handle callback.
            if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                _ = row_position_on_end_tx.send(n_rows);
            }
            if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                _ = n_rows_in_file_tx.send(n_rows);
            }

            return Ok((rx, spawn(TaskPriority::Low, std::future::ready(Ok(())))));
        }

        // Always create a slice. If no slice was given, just make the biggest slice possible.
        let slice_range: Range<usize> = normalized_pre_slice
            .clone()
            .map_or(0..usize::MAX, Range::<usize>::from);

        // Avoid materializing projection info if we are projecting all the columns of this file.
        let projection_indices: Option<Vec<usize>> = if let Some(first_mismatch_idx) =
            (0..file_metadata.schema.len().min(projected_schema.len())).find(|&i| {
                file_metadata.schema.get_at_index(i).unwrap().0
                    != projected_schema.get_at_index(i).unwrap().0
            }) {
            let mut out = Vec::with_capacity(file_metadata.schema.len());

            out.extend(0..first_mismatch_idx);

            out.extend(
                (first_mismatch_idx..projected_schema.len()).filter_map(|i| {
                    file_metadata
                        .schema
                        .index_of(projected_schema.get_at_index(i).unwrap().0)
                }),
            );

            Some(out)
        } else if file_metadata.schema.len() > projected_schema.len() {
            // Names match up to projected schema len.
            Some((0..projected_schema.len()).collect::<Vec<_>>())
        } else {
            // Name order matches up to `file_metadata.schema.len()`, we are projecting all columns
            // in this file.
            None
        };

        if verbose {
            eprintln!(
                "[IpcFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                resolved_pre_slice: {:?} \
                ",
                projection_indices
                    .as_ref()
                    .map_or(file_metadata.schema.len(), |x| x.len()),
                file_metadata.schema.len(),
                pre_slice_arg,
                normalized_pre_slice
            )
        }

        let projection_info: Option<ProjectionInfo> =
            projection_indices.map(|indices| prepare_projection(&file_metadata.schema, indices));
        let projection_info = Arc::new(projection_info);

        // Prepare parameters for Prefetch
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        let record_batch_prefetch_size = self
            .record_batch_prefetch_sync
            .prefetch_limit
            .min(file_metadata.blocks.len())
            .max(1);

        let io_runtime = polars_io::pl_async::get_runtime();
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
            projection_info,
            dictionaries: dictionaries.clone(),
            row_index,
            slice_range: slice_range.clone(),
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
        let metadata = file_metadata.clone();
        let prefetch_task = AbortOnDropHandle(io_runtime.spawn(async move {
            let mut record_batch_data_fetcher = RecordBatchDataFetcher {
                memory_prefetch_func,
                metadata,
                byte_source,
                record_batch_idx: 0,
                prefetch_send,
                rb_prefetch_semaphore,
            };

            // We fetch all record batches so that we know the total number of rows.
            // @TODO: In case of slicing, it would suffice to fetch the record batch
            // headers for any record batch that falls outside of the slice.

            if let Some(rb_prefetch_prev_all_spawned) = rb_prefetch_prev_all_spawned {
                rb_prefetch_prev_all_spawned.wait().await;
            }

            record_batch_data_fetcher.run().await?;

            drop(rb_prefetch_current_all_spawned);

            PolarsResult::Ok(())
        }));

        // Task: Decode.
        let decode_task = AbortOnDropHandle(io_runtime.spawn(async move {
            let mut current_row_offset: IdxSize = 0;

            while let Some((prefetch_task, permit)) = prefetch_recv.recv().await {
                let record_batch_data = prefetch_task.await.unwrap()?;

                // Fetch every record batch so we can determine total row count.
                let rb_num_rows = record_batch_data.num_rows;
                let rb_num_rows =
                    IdxSize::try_from(rb_num_rows).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
                let row_range_end = current_row_offset
                    .checked_add(rb_num_rows)
                    .ok_or(ROW_COUNT_OVERFLOW_ERR)?;
                let row_range = current_row_offset..row_range_end;
                current_row_offset = row_range_end;

                // Only pass to decoder if we need the data.
                if (row_range.start as usize) < slice_range.end {
                    let record_batch_decoder = record_batch_decoder.clone();
                    let decode_fut = async_executor::spawn(TaskPriority::High, async move {
                        record_batch_decoder
                            .record_batch_data_to_df(record_batch_data, row_range)
                            .await
                    });
                    if decode_send.send((decode_fut, permit)).await.is_err() {
                        break;
                    }
                } else {
                    drop(record_batch_data);
                    drop(permit);
                }
            }

            let current_row_offset =
                IdxSize::try_from(current_row_offset).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;

            // Handle callback.
            if let Some(row_position_on_end_tx) = row_position_on_end_tx {
                _ = row_position_on_end_tx.send(current_row_offset);
            }
            if let Some(n_rows_in_file_tx) = n_rows_in_file_tx {
                _ = n_rows_in_file_tx.send(current_row_offset);
            }

            PolarsResult::Ok(())
        }));

        // Task: Distributor.
        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        let last_morsel_min_split = num_pipelines;
        let distribute_task = async_executor::spawn(TaskPriority::High, async move {
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
                    last_morsel_min_split,
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
        let join_task = io_runtime.spawn(async move {
            prefetch_task.await.unwrap()?;
            decode_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        let handle = AbortOnDropHandle(join_task);

        Ok((
            morsel_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }

    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}

impl IpcFileReader {
    /// Total number of rows in the IPC File. This can be slow if the underlying
    /// byte_source is an object store with a high number of Record Batches.
    fn _n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        let InitializedState {
            file_metadata,
            byte_source,
            dictionaries: _,
            n_rows_in_file,
        } = self.init_data.as_mut().unwrap();

        if n_rows_in_file.is_none() {
            match &**byte_source {
                DynByteSource::MemSlice(MemSliceByteSource(memslice)) => {
                    let n_rows: i64 = get_row_count_from_blocks(
                        &mut std::io::Cursor::new(memslice.as_ref()),
                        &file_metadata.blocks,
                    )?;

                    let n_rows = IdxSize::try_from(n_rows)
                        .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

                    *n_rows_in_file = Some(n_rows);
                },
                byte_source @ DynByteSource::Cloud(_) => {
                    let io_runtime = polars_io::pl_async::get_runtime();

                    let mut n_rows = 0;
                    let mut message_scratch = Vec::new();
                    let mut ranges: Vec<_> = file_metadata
                        .blocks
                        .iter()
                        .map(|block| {
                            block.offset as usize
                                ..block.offset as usize + block.meta_data_length as usize
                        })
                        .collect();

                    let bytes_map = io_runtime.block_on(byte_source.get_ranges(&mut ranges))?;
                    assert_eq!(bytes_map.len(), ranges.len());

                    for bytes in bytes_map.into_values() {
                        let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));
                        n_rows += reader.record_batch_num_rows(&mut message_scratch)?;
                    }

                    let n_rows = IdxSize::try_from(n_rows)
                        .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

                    *n_rows_in_file = Some(n_rows);
                },
            }
        }

        Ok(n_rows_in_file.unwrap())
    }

    /// Retrieve file metadata from the source.
    async fn fetch_file_metadata(&self) -> PolarsResult<Arc<FileMetadata>> {
        if self.verbose {
            eprintln!("[IpcFileReader]: fetching file metadata");
        }

        let metadata = match self.scan_source.as_scan_source_ref() {
            ScanSourceRef::Path(path) => match path {
                PlPathRef::Cloud(_) => {
                    feature_gated!("cloud", {
                        get_runtime().block_in_place_on(async {
                            let metadata: PolarsResult<_> = {
                                let reader = polars_io::ipc::IpcReaderAsync::from_uri(
                                    path,
                                    self.cloud_options.as_deref(),
                                )
                                .await?;

                                Ok(reader.metadata().await?)
                            };
                            metadata
                        })?
                    })
                },
                PlPathRef::Local(path) => {
                    // Local file I/O is typically synchronous in Arrow-rs
                    let mut reader = std::io::BufReader::new(polars_utils::open_file(path)?);
                    read_file_metadata(&mut reader)?
                },
            },
            ScanSourceRef::File(file) => {
                let mut reader = std::io::BufReader::new(file);
                read_file_metadata(&mut reader)?
            },
            ScanSourceRef::Buffer(buff) => {
                let mut reader = std::io::Cursor::new(buff);
                read_file_metadata(&mut reader)?
            },
        };

        Ok(Arc::new(metadata))
    }

    fn _row_position_after_slice(&mut self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        Ok(calc_row_position_after_slice(
            self._n_rows_in_file()?,
            pre_slice,
        ))
    }
}

async fn read_dictionaries(
    byte_source: &DynByteSource,
    file_metadata: Arc<FileMetadata>,
    verbose: bool,
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
        )?;
    }

    Ok(dictionaries)
}
