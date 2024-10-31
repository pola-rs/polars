use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType};
use polars_core::scalar::Scalar;
use polars_core::utils::arrow::array::TryExtend;
use polars_core::utils::arrow::io::ipc::read::{read_file_metadata, FileMetadata, FileReader};
use polars_error::PolarsResult;
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
use polars_utils::IdxSize;

use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::{
    ComputeNode, JoinHandle, Morsel, MorselSeq, PortState, TaskPriority, TaskScope,
};
use crate::pipe::{RecvPort, SendPort};
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;

pub struct IpcSourceNodeConfig {
    row_index: Option<RowIndex>,
    projection: Option<Vec<usize>>,
    slice: Option<(i64, usize)>,

    rechunk: bool,
    include_file_paths: Option<PlSmallStr>,

    first_metadata: FileMetadata,
}

pub struct IpcSourceNode {
    sources: ScanSources,

    config: IpcSourceNodeConfig,

    /// Can the SendPort be closed?
    is_finished: bool,
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
        mut first_metadata: Option<FileMetadata>,
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
                let Some(source) = sources.iter().next() else {
                    todo!()
                };

                let source = source.to_memslice()?;
                read_file_metadata(&mut std::io::Cursor::new(&*source))?
            },
        };

        let projection = with_columns
            .as_ref()
            .map(|cols| columns_to_projection(cols, &first_metadata.schema))
            .transpose()?;

        Ok(IpcSourceNode {
            sources,

            config: IpcSourceNodeConfig {
                row_index,
                projection,
                slice,

                rechunk,
                include_file_paths,

                first_metadata,
            },

            is_finished: false,
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

impl ComputeNode for IpcSourceNode {
    fn name(&self) -> &str {
        "ipc_source"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self.is_finished {
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
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        struct BatchMessage {
            source_idx: usize,
            memslice: Arc<MemSlice>,
            metadata: Arc<FileMetadata>,
            file_path: Option<Arc<str>>,
            row_idx_offset: IdxSize,
            slice: Range<usize>,
            block_range: Range<usize>,
            seq: u64,
        }

        let senders = send[0].take().unwrap().parallel();
        let num_senders = senders.len();
        let (mut distribute, rxs) =
            distributor_channel::<BatchMessage>(num_senders, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

        let config = &self.config;
        let sources = &self.sources;
        let is_finished = &mut self.is_finished;

        let decoder_tasks = senders
            .into_iter()
            .zip(rxs)
            .map(|(mut send, mut rx)| {
                scope.spawn_task(TaskPriority::Low, async move {
                    let wait_group = WaitGroup::default();
                    let source_token = SourceToken::new();

                    let mut reader_source_idx = usize::MAX;
                    let mut reader = FileReader::new(
                        Cursor::new(MemSlice::default()),
                        config.first_metadata.clone(),
                        config.projection.clone(),
                        None,
                    );

                    let schema = reader.schema();
                    let pl_schema = schema
                        .iter()
                        .map(|(n, f)| (n.clone(), DataType::from_arrow(&f.dtype, true)))
                        .collect();

                    while let Ok(m) = rx.recv().await {
                        let BatchMessage {
                            source_idx,
                            memslice: source,
                            metadata,
                            file_path,
                            row_idx_offset,
                            slice,
                            seq,
                            block_range,
                        } = m;

                        // Update the reader if we moved onto a different source. This allows
                        // us to amortized allocations.
                        if reader_source_idx != source_idx {
                            reader_source_idx = source_idx;
                            reader.update_file(
                                Cursor::new(source.as_ref().clone()),
                                metadata.as_ref().clone(),
                            );
                        }

                        // Create the DataFrame with the appropriate schema and append all the record
                        // batches to it. This will perform schema validation as well.
                        let mut df = DataFrame::empty_with_schema(&pl_schema);
                        reader.set_current_block(block_range.start);
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

                        let mut morsel = Morsel::new(df, MorselSeq::new(seq), source_token.clone());
                        morsel.set_consume_token(wait_group.token());
                        if send.send(morsel).await.is_err() {
                            break;
                        }

                        wait_group.wait().await;
                        if source_token.stop_requested() {
                            break;
                        }
                    }

                    PolarsResult::Ok(())
                })
            })
            .collect::<Vec<_>>();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            struct Batch {
                row_idx_offset: IdxSize,
                block_start: usize,
                num_rows: usize,
            }

            let mut seq = 0;
            let mut row_idx_offset = config.row_index.as_ref().map_or(0, |ri| ri.offset);

            // Always create a slice. If no slice was given, just make the biggest slice possible.
            let mut slice = config.slice.map_or(0..usize::MAX, |(offset, length)| {
                let offset = offset as usize;
                offset..offset + length
            });

            // Batch completion parameters
            let batch_size_limit = get_ideal_morsel_size();
            let sliced_batch_size_limit = slice.len().div_ceil(num_senders);
            let batch_block_limit = if sources.len() >= num_senders {
                // If there are more files than decoder tasks, try to subdivide the files instead
                // of the blocks.
                usize::MAX
            } else {
                config.first_metadata.blocks.len().div_ceil(num_senders)
            };

            // Amortize allocations
            let mut data_scratch = Vec::new();
            let mut message_scratch = Vec::new();

            for (source_idx, source) in sources.iter().enumerate() {
                if slice.is_empty() {
                    break;
                }

                let memslice = Arc::new(source.to_memslice()?);

                let metadata = if source_idx == 0 {
                    config.first_metadata.clone()
                } else {
                    read_file_metadata(&mut std::io::Cursor::new(memslice.as_ref().as_ref()))?
                };
                let metadata_arc = Arc::new(metadata.clone());

                let file_path: Option<Arc<str>> = config
                    .include_file_paths
                    .as_ref()
                    .map(|_| source.to_include_path_name().into());

                let mut reader = FileReader::new(
                    Cursor::new(memslice.as_ref()),
                    metadata,
                    config.projection.clone(),
                    None,
                );

                reader.set_scratches((
                    std::mem::take(&mut data_scratch),
                    std::mem::take(&mut message_scratch),
                ));

                if slice.start > 0 {
                    // Skip over all blocks that the slice would skip anyway.
                    let new_offset = reader.skip_blocks_till_limit(slice.start as u64)?;

                    row_idx_offset += (slice.start as u64 - new_offset) as IdxSize;
                    slice = new_offset as usize..new_offset as usize + slice.len();

                    // If we skip the entire file. Don't even try to read from it.
                    if reader.get_current_block() == reader.metadata().blocks.len() {
                        (data_scratch, message_scratch) = reader.get_scratches();
                        continue;
                    }
                }

                let mut batch = Batch {
                    row_idx_offset,
                    block_start: reader.get_current_block(),
                    num_rows: 0,
                };

                while !slice.is_empty() {
                    let mut is_batch_complete = false;

                    match reader.next_record_batch() {
                        // If we have no more record batches available, we want to send what is
                        // left.
                        None => is_batch_complete = true,
                        Some(record_batch) => {
                            let rb_num_rows = record_batch?.length()? as usize;
                            batch.num_rows += rb_num_rows;
                            row_idx_offset += rb_num_rows as IdxSize;
                        },
                    }

                    let current_block = reader.get_current_block();

                    // Subdivide into batches for large files
                    is_batch_complete |= batch.num_rows >= batch_size_limit;
                    // Subdivide into batches if the file is sliced
                    is_batch_complete |= batch.num_rows >= sliced_batch_size_limit;
                    // Subdivide into batches for small files
                    is_batch_complete |= current_block - batch.block_start >= batch_block_limit;

                    // Batch blocks such that we send appropriately sized morsels. We guarantee a
                    // lower bound here, but not an upperbound.
                    if is_batch_complete {
                        let batch_slice = slice_take(&mut slice, batch.num_rows);
                        let block_range = batch.block_start..current_block;

                        let message = BatchMessage {
                            source_idx,
                            memslice: memslice.clone(),
                            metadata: metadata_arc.clone(),
                            file_path: file_path.clone(),
                            row_idx_offset: batch.row_idx_offset,
                            slice: batch_slice,
                            seq,
                            block_range,
                        };

                        if distribute.send(message).await.is_err() {
                            break;
                        };

                        batch = Batch {
                            row_idx_offset,
                            block_start: current_block,
                            num_rows: 0,
                        };
                        seq += 1;
                    }
                }

                (data_scratch, message_scratch) = reader.get_scratches();
            }

            drop(distribute); // Inform decoder tasks to stop
            for decoder_task in decoder_tasks {
                decoder_task.await?;
            }
            *is_finished = true; // Inform the graph that the port is done.

            PolarsResult::Ok(())
        }));
    }
}
