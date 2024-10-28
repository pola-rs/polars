use std::io::Cursor;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, ArrowField, ArrowSchema, Column, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::series::Series;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::io::ipc::read::{read_file_metadata, FileMetadata, FileReader};
use polars_error::PolarsResult;
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::utils::{apply_projection, columns_to_projection};
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

pub struct IpcSourceNode {
    sources: ScanSources,

    row_index: Option<RowIndex>,
    projection: Option<Vec<usize>>,
    slice: Option<(i64, usize)>,

    rechunk: bool,
    include_file_paths: Option<PlSmallStr>,

    projected_schema: Arc<ArrowSchema>,

    /// Can the SendPort be closed?
    is_finished: AtomicBool,

    first_metadata: FileMetadata,
}

impl IpcSourceNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,
        _file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: IpcScanOptions,
        _cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        mut first_metadata: Option<FileMetadata>,
    ) -> PolarsResult<Self> {
        assert!(predicate.is_none()); // This should have been removed during lower_ir

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
            allow_missing_columns,
        } = file_options;

        // @TODO: All the things the IPC source does not support yet.
        if hive_parts.is_some() || sources.is_cloud_url() || allow_missing_columns {
            todo!();
        }

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

        let projected_schema = projection.as_ref().map_or_else(
            || first_metadata.schema.clone(),
            |prj| Arc::new(apply_projection(&first_metadata.schema, prj)),
        );

        Ok(IpcSourceNode {
            sources,

            row_index,
            projection,
            slice,

            rechunk,
            include_file_paths,

            projected_schema,

            is_finished: AtomicBool::new(false),

            first_metadata,
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

        let is_finished = self.is_finished.load(Ordering::Relaxed);
        if is_finished {
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

        let senders = send[0].take().unwrap().parallel();

        let slf = &*self;

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            struct BatchMessage {
                source_idx: usize,
                memslice: Arc<MemSlice>,
                metadata: Arc<FileMetadata>,
                file_path: Option<Arc<str>>,
                row_start: IdxSize,
                slice: Range<usize>,
                block_range: Range<usize>,
                seq: u64,
            }

            let num_senders = senders.len();
            let (mut distribute, rxs) =
                distributor_channel::<BatchMessage>(num_senders, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);

            let decoding_tasks = senders
                .into_iter()
                .zip(rxs)
                .map(|(mut send, mut rx)| {
                    scope.spawn_task(TaskPriority::Low, async move {
                        let wait_group = WaitGroup::default();
                        let source_token = SourceToken::new();

                        let mut reader_source_idx = usize::MAX;
                        let mut reader = FileReader::new(
                            Cursor::new(MemSlice::default()),
                            slf.first_metadata.clone(),
                            slf.projection.clone(),
                            None,
                        );

                        let schema = reader.schema();
                        let fields = slf
                            .projected_schema
                            .iter()
                            .map(|(name, field)| {
                                ArrowField::new(name.clone(), field.dtype.clone(), true)
                            })
                            .collect::<Vec<_>>();
                        let mut columns: Vec<Vec<Box<dyn Array>>> =
                            Vec::with_capacity(schema.len());

                        while let Ok(m) = rx.recv().await {
                            let BatchMessage {
                                source_idx,
                                memslice: source,
                                metadata,
                                file_path,
                                row_start,
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

                            assert!(!block_range.is_empty());

                            reader.set_current_block(block_range.start);

                            let record_batch = reader.next().unwrap()?;

                            let schema = reader.schema();
                            assert_eq!(record_batch.arrays().len(), schema.len());

                            let mut height = record_batch.len();

                            columns.clear();
                            columns.extend(record_batch.into_arrays().into_iter().map(|v| vec![v]));

                            for _ in block_range.into_iter().skip(1) {
                                let record_batch = reader.next().unwrap()?;

                                assert_eq!(record_batch.arrays().len(), reader.schema().len());

                                height += record_batch.len();

                                if cfg!(debug_assertions) {
                                    record_batch
                                        .arrays()
                                        .iter()
                                        .all(|a| a.len() == record_batch.len());
                                }

                                for (column, array) in
                                    columns.iter_mut().zip(record_batch.into_arrays())
                                {
                                    column.push(array);
                                }
                            }

                            let df_cols = columns
                                .iter_mut()
                                .zip(fields.iter())
                                .map(|(chunks, field)| {
                                    let mut series =
                                        Series::try_from((field, std::mem::take(chunks)))?;

                                    if slf.rechunk {
                                        series = series.rechunk();
                                    }

                                    Ok(series.into_column())
                                })
                                .collect::<PolarsResult<Vec<Column>>>()?;
                            let mut df = DataFrame::new(df_cols)?;

                            assert_eq!(df.height(), height);

                            df = df.slice(slice.start as i64, slice.len());

                            if let Some(RowIndex { name, offset }) = &slf.row_index {
                                df = df.with_row_index(
                                    name.clone(),
                                    Some(offset + row_start + slice.start as IdxSize),
                                )?;
                            }

                            if let Some(col) = slf.include_file_paths.as_ref() {
                                let file_path = file_path.unwrap();
                                df.with_column(Column::new_scalar(
                                    col.clone(),
                                    Scalar::new(
                                        DataType::String,
                                        AnyValue::StringOwned(PlSmallStr::from(file_path.as_ref())),
                                    ),
                                    df.height(),
                                ))?;
                            }

                            let mut morsel =
                                Morsel::new(df, MorselSeq::new(seq), source_token.clone());
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

            let mut slice = slf.slice.map_or(0..usize::MAX, |(offset, length)| {
                if offset < 0 {
                    todo!();
                }

                let offset = offset as usize;

                offset..offset + length
            });

            let mut seq = 0;

            let ideal_morsel_size = get_ideal_morsel_size();

            let max_blocks_per_sender = if slf.sources.len() >= num_senders {
                usize::MAX
            } else {
                slf.first_metadata.blocks.len().div_ceil(num_senders)
            };

            struct Batch {
                row_start: IdxSize,
                block_start: usize,
                length: usize,
            }

            let mut row_start = 0;

            let ideal_sliced_batch_length = slice.len().div_ceil(num_senders);

            // Amortize allocations
            let mut data_scratch = Vec::new();
            let mut message_scratch = Vec::new();

            for (source_idx, source) in slf.sources.iter().enumerate() {
                if slice.is_empty() {
                    break;
                }

                let memslice = Arc::new(source.to_memslice()?);
                let metadata = if source_idx == 0 {
                    slf.first_metadata.clone()
                } else {
                    read_file_metadata(&mut std::io::Cursor::new(memslice.as_ref().as_ref()))?
                };

                let file_path: Option<Arc<str>> = slf
                    .include_file_paths
                    .as_ref()
                    .map(|_| source.to_include_path_name().into());
                let metadata_arc = Arc::new(metadata.clone());

                let mut reader = FileReader::new(
                    Cursor::new(memslice.as_ref()),
                    metadata,
                    slf.projection.clone(),
                    None,
                );

                reader.set_scratches((
                    std::mem::take(&mut data_scratch),
                    std::mem::take(&mut message_scratch),
                ));

                if slice.start > 0 {
                    // Skip over all blocks that the slice would skip anyway.
                    let new_offset = reader.skip_blocks_till_limit(slice.start as u64)?;

                    row_start += (slice.start as u64 - new_offset) as IdxSize;
                    slice = new_offset as usize..new_offset as usize + slice.len();

                    // If we skip the entire file. Don't even try to read from it.
                    if reader.get_current_block() == reader.metadata().blocks.len() {
                        (data_scratch, message_scratch) = reader.get_scratches();
                        continue;
                    }
                }

                let mut batch = Batch {
                    row_start,
                    block_start: reader.get_current_block(),
                    length: 0,
                };

                while !slice.is_empty() {
                    let Some(record_batch) = reader.next_record_batch() else {
                        break;
                    };

                    let record_batch = record_batch?;
                    let record_batch_len = record_batch.length()? as usize;
                    let block_end = reader.get_current_block();

                    batch.length += record_batch_len;

                    let mut is_batch_complete = false;

                    // Subdivide into batches if the file is sliced
                    is_batch_complete |= batch.length >= ideal_sliced_batch_length;
                    // Subdivide into batches for large files
                    is_batch_complete |= batch.length >= ideal_morsel_size;
                    // Subdivide into batches for small files
                    is_batch_complete |= block_end - batch.block_start >= max_blocks_per_sender;

                    row_start += record_batch_len as IdxSize;

                    // Batch blocks such that we send appropriately sized morsels. We guarantee a
                    // lower bound here, but not an upperbound.
                    if is_batch_complete {
                        let batch_slice = slice_take(&mut slice, batch.length);
                        let block_range = batch.block_start..block_end;

                        let message = BatchMessage {
                            source_idx,
                            memslice: memslice.clone(),
                            metadata: metadata_arc.clone(),
                            file_path: file_path.clone(),
                            row_start: batch.row_start,
                            slice: batch_slice,
                            seq,
                            block_range,
                        };

                        if distribute.send(message).await.is_err() {
                            break;
                        };

                        batch = Batch {
                            row_start,
                            block_start: block_end,
                            length: 0,
                        };
                        seq += 1;
                    }
                }

                // If we still have a last batch to send, just try to send it.
                if batch.length > 0 {
                    let batch_slice = slice_take(&mut slice, batch.length);
                    let block_range = batch.block_start..reader.get_current_block();

                    let message = BatchMessage {
                        source_idx,
                        memslice: memslice.clone(),
                        metadata: metadata_arc.clone(),
                        file_path,
                        row_start,
                        slice: batch_slice,
                        seq,
                        block_range,
                    };

                    _ = distribute.send(message).await;
                }

                (data_scratch, message_scratch) = reader.get_scratches();
            }

            // Dropping the sender channel will stop waiting decoder tasks
            drop(distribute);

            for decoding_task in decoding_tasks {
                decoding_task.await?;
            }

            // Inform the graph that the port is done.
            slf.is_finished.store(true, Ordering::Relaxed);

            PolarsResult::Ok(())
        }));
    }
}
