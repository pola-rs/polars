use std::cmp::Reverse;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType};
use polars_core::scalar::Scalar;
use polars_core::utils::arrow::array::TryExtend;
use polars_core::utils::arrow::io::ipc::read::{
    prepare_projection, read_file_metadata, FileMetadata, FileReader, ProjectionInfo,
};
use polars_error::{ErrString, PolarsError, PolarsResult};
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::utils::columns_to_projection;
use polars_io::RowIndex;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{FileInfo, ScanSources};
use polars_plan::prelude::FileScanOptions;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::IdxSize;

use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::{
    ComputeNode, JoinHandle, Morsel, MorselSeq, PortState, TaskPriority, TaskScope,
};
use crate::pipe::{RecvPort, SendPort};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (polars-u64-idx package on python)",
));

pub struct IpcSourceNode {
    sources: ScanSources,

    config: IpcSourceNodeConfig,
    num_pipelines: usize,

    /// Every phase we need to be able to continue from where we left off, so we save the state of
    /// the Walker task.
    state: IpcSourceNodeState,
}

pub struct IpcSourceNodeConfig {
    row_index: Option<RowIndex>,
    projection_info: Option<ProjectionInfo>,

    rechunk: bool,
    include_file_paths: Option<PlSmallStr>,

    first_metadata: Arc<FileMetadata>,
}

pub struct IpcSourceNodeState {
    morsel_seq: u64,
    row_idx_offset: IdxSize,

    slice: Range<usize>,

    source_idx: usize,
    source: Option<Source>,
}

pub struct Source {
    file_path: Option<Arc<str>>,

    memslice: Arc<MemSlice>,
    metadata: Arc<FileMetadata>,

    block_offset: usize,
}

impl IpcSourceNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,
        _file_info: FileInfo,
        _hive_parts: Option<Arc<Vec<HivePartitions>>>, // @TODO
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: IpcScanOptions,
        _cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        mut first_metadata: Option<Arc<FileMetadata>>,
    ) -> PolarsResult<Self> {
        // These should have all been removed during lower_ir
        assert!(predicate.is_none());
        assert!(!sources.is_empty());

        let IpcScanOptions = options;

        let FileScanOptions {
            slice,
            with_columns,
            cache: _, // @TODO
            row_index,
            rechunk,
            file_counter: _, // @TODO
            hive_options: _, // @TODO
            glob: _,         // @TODO
            include_file_paths,
            allow_missing_columns: _, // @TODO
        } = file_options;

        let first_metadata = match first_metadata.take() {
            Some(md) => md,
            None => {
                let source = sources.iter().next().unwrap();
                let source = source.to_memslice()?;
                Arc::new(read_file_metadata(&mut std::io::Cursor::new(&*source))?)
            },
        };

        let projection = with_columns
            .as_ref()
            .map(|cols| columns_to_projection(cols, &first_metadata.schema))
            .transpose()?;
        let projection_info = projection
            .as_ref()
            .map(|p| prepare_projection(&first_metadata.schema, p.clone()));

        let state = IpcSourceNodeState {
            morsel_seq: 0,
            row_idx_offset: row_index.as_ref().map_or(0, |ri| ri.offset),

            // Always create a slice. If no slice was given, just make the biggest slice possible.
            slice: slice.map_or(0..usize::MAX, |(offset, length)| {
                let offset = offset as usize;
                offset..offset + length
            }),

            source_idx: 0,
            source: None,
        };

        Ok(IpcSourceNode {
            sources,

            config: IpcSourceNodeConfig {
                row_index,
                projection_info,

                rechunk,
                include_file_paths,

                first_metadata,
            },

            num_pipelines: 0,

            state,
        })
    }
}

/// Move `slice` forward by `n` and return the slice until then.
fn slice_take(slice: &mut Range<usize>, n: usize) -> Range<usize> {
    let offset = slice.start;
    let length = slice.len();

    assert!(offset < n);

    let chunk_length = (n - offset).min(length);
    let rng = offset..offset + chunk_length;
    *slice = 0..length - chunk_length;

    rng
}

fn get_max_morsel_size() -> usize {
    std::env::var("POLARS_STREAMING_IPC_SOURCE_MAX_MORSEL_SIZE")
        .map_or_else(
            |_| get_ideal_morsel_size(),
            |v| {
                v.parse::<usize>().expect(
                    "POLARS_STREAMING_IPC_SOURCE_MAX_MORSEL_SIZE does not contain valid size",
                )
            },
        )
        .max(1)
}

impl ComputeNode for IpcSourceNode {
    fn name(&self) -> &str {
        "ipc_source"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.num_pipelines = num_pipelines;
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self.state.slice.is_empty() || self.state.source_idx >= self.sources.len() {
            send[0] = PortState::Done;
        }

        if send[0] != PortState::Done {
            send[0] = PortState::Ready;
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
        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);

        // Split size for morsels.
        let max_morsel_size = get_max_morsel_size();
        let source_token = SourceToken::new();

        let num_pipelines = self.num_pipelines;
        let config = &self.config;
        let sources = &self.sources;
        let state = &mut self.state;

        /// Messages sent from Walker task to Decoder tasks.
        struct BatchMessage {
            memslice: Arc<MemSlice>,
            metadata: Arc<FileMetadata>,
            file_path: Option<Arc<str>>,
            row_idx_offset: IdxSize,
            slice: Range<usize>,
            block_range: Range<usize>,
            morsel_seq_base: u64,
        }

        // Walker task -> Decoder tasks.
        let (mut batch_tx, batch_rxs) =
            distributor_channel::<BatchMessage>(num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        // Decoder tasks -> Distributor task.
        let (mut decoded_rx, decoded_tx) = Linearizer::<Priority<Reverse<MorselSeq>, Morsel>>::new(
            num_pipelines,
            DEFAULT_LINEARIZER_BUFFER_SIZE,
        );
        // Distributor task -> output.
        let mut sender = send_ports[0].take().unwrap().serial();

        // Distributor task.
        //
        // Shuffles morsels from `n` producers amongst `n` consumers.
        //
        // If record batches in the source IPC file are large, one decoder might produce many
        // morsels at the same time. At the same time, other decoders might not produce anything.
        // Therefore, we would like to distribute the output of a single decoder task over the
        // available output pipelines.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Some(morsel) = decoded_rx.get().await {
                if sender.send(morsel.1).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Decoder tasks.
        //
        // Tasks a IPC file and certain number of blocks and decodes each block as a record batch.
        // Then, all record batches are concatenated into a DataFrame. If the resulting DataFrame
        // is too large, which happens when we have one very large block, the DataFrame is split
        // into smaller pieces an spread among the pipelines.
        let decoder_tasks = decoded_tx.into_iter().zip(batch_rxs)
            .map(|(mut send, mut rx)| {
                let source_token = source_token.clone();
                scope.spawn_task(TaskPriority::Low, async move {
                    // Amortize allocations.
                    let mut data_scratch = Vec::new();
                    let mut message_scratch = Vec::new();
                    let mut projection_info = config.projection_info.clone();

                    let schema = projection_info.as_ref().map_or(config.first_metadata.schema.as_ref(), |ProjectionInfo { schema, .. }| schema);
                    let pl_schema = schema
                        .iter()
                        .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
                        .collect();

                    while let Ok(m) = rx.recv().await {
                        let BatchMessage {
                            memslice: source,
                            metadata,
                            file_path,
                            row_idx_offset,
                            slice,
                            morsel_seq_base,
                            block_range,
                        } = m;

                        let mut reader = FileReader::new_with_projection_info(
                            Cursor::new(source.as_ref()),
                            metadata.as_ref().clone(),
                            std::mem::take(&mut projection_info),
                            None,
                        );
                        reader.set_current_block(block_range.start);
                        reader.set_scratches((
                            std::mem::take(&mut data_scratch),
                            std::mem::take(&mut message_scratch),
                        ));

                        // Create the DataFrame with the appropriate schema and append all the record
                        // batches to it. This will perform schema validation as well.
                        let mut df = DataFrame::empty_with_schema(&pl_schema);
                        df.try_extend(reader.by_ref().take(block_range.len()))?;

                        df = df.slice(slice.start as i64, slice.len());

                        if config.rechunk {
                            df.rechunk_mut();
                        }

                        if let Some(RowIndex { name, offset: _ }) = &config.row_index {
                            let offset = row_idx_offset + slice.start as IdxSize;
                            df = df.with_row_index(name.clone(), Some(offset))?;
                        }

                        if let Some(col) = config.include_file_paths.as_ref() {
                            let file_path = file_path.unwrap();
                            let file_path = Scalar::from(PlSmallStr::from(file_path.as_ref()));
                            df.with_column(Column::new_scalar(
                                col.clone(),
                                file_path,
                                df.height(),
                            ))?;
                        }

                        // If the block is very large, we want to split the block amongst the
                        // pipelines. That will at least allow some parallelism.
                        if df.height() > max_morsel_size && config::verbose() {
                            eprintln!("IPC source encountered a (too) large record batch of {} rows. Splitting and continuing.", df.height());
                        }
                        for i in 0..df.height().div_ceil(max_morsel_size) {
                            let morsel = df.slice((i * max_morsel_size) as i64, max_morsel_size);
                            let seq = MorselSeq::new(morsel_seq_base + i as u64);
                            let morsel = Morsel::new(
                                morsel,
                                seq,
                                source_token.clone(),
                            );
                            if send.insert(Priority(Reverse(seq), morsel)).await.is_err() {
                                break;
                            }
                        }

                        (data_scratch, message_scratch) = reader.take_scratches();
                        projection_info = reader.take_projection_info();
                    }

                    PolarsResult::Ok(())
                })
            })
            .collect::<Vec<_>>();

        // Walker task.
        //
        // Walks all the sources and supplies block ranges to the decoder tasks.
        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            struct Batch {
                row_idx_offset: IdxSize,
                block_start: usize,
                num_rows: usize,
            }

            // Batch completion parameters
            let batch_size_limit = get_ideal_morsel_size();
            let sliced_batch_size_limit = state.slice.len().div_ceil(num_pipelines);
            let batch_block_limit = if sources.len() >= num_pipelines {
                // If there are more files than decoder tasks, try to subdivide the files instead
                // of the blocks.
                usize::MAX
            } else {
                config.first_metadata.blocks.len().div_ceil(num_pipelines)
            };

            // Amortize allocations
            let mut data_scratch = Vec::new();
            let mut message_scratch = Vec::new();
            let mut projection_info = config.projection_info.clone();

            'source_loop: while !state.slice.is_empty() {
                let source = match state.source {
                    Some(ref mut source) => source,
                    None => {
                        let Some(source) = sources.get(state.source_idx) else {
                            break;
                        };

                        let file_path: Option<Arc<str>> = config
                            .include_file_paths
                            .as_ref()
                            .map(|_| source.to_include_path_name().into());
                        let memslice = source.to_memslice()?;
                        let metadata = if state.source_idx == 0 {
                            config.first_metadata.clone()
                        } else {
                            Arc::new(read_file_metadata(&mut std::io::Cursor::new(
                                memslice.as_ref(),
                            ))?)
                        };

                        state.source.insert(Source {
                            file_path,
                            memslice: Arc::new(memslice),
                            metadata,
                            block_offset: 0,
                        })
                    },
                };

                let mut reader = FileReader::new_with_projection_info(
                    Cursor::new(source.memslice.as_ref()),
                    source.metadata.as_ref().clone(),
                    std::mem::take(&mut projection_info),
                    None,
                );
                reader.set_current_block(source.block_offset);
                reader.set_scratches((
                    std::mem::take(&mut data_scratch),
                    std::mem::take(&mut message_scratch),
                ));

                if state.slice.start > 0 {
                    // Skip over all blocks that the slice would skip anyway.
                    let new_offset = reader.skip_blocks_till_limit(state.slice.start as u64)?;

                    state.row_idx_offset += (state.slice.start as u64 - new_offset) as IdxSize;
                    state.slice = new_offset as usize..new_offset as usize + state.slice.len();

                    // If we skip the entire file. Don't even try to read from it.
                    if reader.get_current_block() == reader.metadata().blocks.len() {
                        (data_scratch, message_scratch) = reader.take_scratches();
                        projection_info = reader.take_projection_info();
                        state.source.take();
                        state.source_idx += 1;
                        continue;
                    }
                }

                let mut batch = Batch {
                    row_idx_offset: state.row_idx_offset,
                    block_start: reader.get_current_block(),
                    num_rows: 0,
                };

                // We don't yet want to commit these values to the state in case this batch gets
                // cancelled.
                let mut uncommitted_slice = state.slice.clone();
                let mut uncommitted_row_idx_offset = state.row_idx_offset;
                while !state.slice.is_empty() {
                    let mut is_batch_complete = false;

                    match reader.next_record_batch() {
                        None if batch.num_rows == 0 => break,

                        // If we have no more record batches available, we want to send what is
                        // left.
                        None => is_batch_complete = true,
                        Some(record_batch) => {
                            let rb_num_rows = record_batch?.length()? as usize;
                            batch.num_rows += rb_num_rows;

                            // We need to ensure that we are not overflowing the IdxSize maximum
                            // capacity.
                            let rb_num_rows = IdxSize::try_from(rb_num_rows)
                                .map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
                            uncommitted_row_idx_offset = uncommitted_row_idx_offset
                                .checked_add(rb_num_rows)
                                .ok_or(ROW_COUNT_OVERFLOW_ERR)?;
                        },
                    }

                    let current_block = reader.get_current_block();

                    // Subdivide into batches for large files.
                    is_batch_complete |= batch.num_rows >= batch_size_limit;
                    // Subdivide into batches if the file is sliced.
                    is_batch_complete |= batch.num_rows >= sliced_batch_size_limit;
                    // Subdivide into batches for small files.
                    is_batch_complete |= current_block - batch.block_start >= batch_block_limit;

                    // Batch blocks such that we send appropriately sized morsels. We guarantee a
                    // lower bound here, but not an upper bound.
                    if is_batch_complete {
                        let batch_slice = slice_take(&mut uncommitted_slice, batch.num_rows);
                        let batch_slice_len = batch_slice.len();
                        let block_range = batch.block_start..current_block;

                        let message = BatchMessage {
                            memslice: source.memslice.clone(),
                            metadata: source.metadata.clone(),
                            file_path: source.file_path.clone(),
                            row_idx_offset: batch.row_idx_offset,
                            slice: batch_slice,
                            morsel_seq_base: state.morsel_seq,
                            block_range,
                        };

                        if source_token.stop_requested() {
                            break 'source_loop;
                        }

                        if batch_tx.send(message).await.is_err() {
                            // This should only happen if the receiver of the decoder
                            // has broken off, meaning no further input will be needed.
                            break 'source_loop;
                        }

                        // Commit the changes to the state.
                        // Now, we know that the a decoder will process it.
                        //
                        // This might generate several morsels if the record batch is very large.
                        state.morsel_seq += batch_slice_len.div_ceil(max_morsel_size) as u64;
                        state.slice = uncommitted_slice.clone();
                        state.row_idx_offset = uncommitted_row_idx_offset;
                        source.block_offset = current_block;

                        batch = Batch {
                            row_idx_offset: state.row_idx_offset,
                            block_start: current_block,
                            num_rows: 0,
                        };
                    }
                }

                (data_scratch, message_scratch) = reader.take_scratches();
                projection_info = reader.take_projection_info();

                state.source.take();
                state.source_idx += 1;
            }

            drop(batch_tx); // Inform decoder tasks to stop.
            for decoder_task in decoder_tasks {
                decoder_task.await?;
            }

            PolarsResult::Ok(())
        }));
    }
}
