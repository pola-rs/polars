use std::cmp::Reverse;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaExt, SchemaRef};
use polars_core::utils::arrow::array::TryExtend;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_core::utils::arrow::io::ipc::read::{
    get_row_count_from_blocks, prepare_projection, read_file_metadata, FileMetadata, FileReader,
    ProjectionInfo,
};
use polars_core::utils::slice_offsets;
use polars_error::{ErrString, PolarsError, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::utils::columns_to_projection;
use polars_io::RowIndex;
use polars_plan::plans::{FileInfo, ScanSource, ScanSources};
use polars_plan::prelude::FileScanOptions;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::IdxSize;

use super::multi_scan::{MultiScanable, RowRestrication};
use super::{SourceNode, SourceOutput};
use crate::async_executor::spawn;
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::{get_ideal_morsel_size, SourceToken};
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (polars-u64-idx package on python)",
));

pub struct IpcSourceNode {
    source: Source,

    row_index: Option<RowIndex>,
    slice: Range<usize>,

    projection_info: Option<ProjectionInfo>,

    rechunk: bool,
    include_file_paths: Option<PlSmallStr>,
}

#[derive(Clone)]
pub struct Source {
    file_path: Option<Arc<str>>,
    memslice: Arc<MemSlice>,
    metadata: Arc<FileMetadata>,
}

impl IpcSourceNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,
        _file_info: FileInfo,
        options: IpcScanOptions,
        _cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        mut metadata: Option<Arc<FileMetadata>>,
    ) -> PolarsResult<Self> {
        assert!(!sources.is_empty());
        assert_eq!(sources.len(), 1);

        let IpcScanOptions = options;

        let FileScanOptions {
            slice,
            with_columns,
            cache: _, // @TODO
            row_index,
            rechunk,
            file_counter: _,
            hive_options: _,
            glob: _,
            include_file_paths,
            allow_missing_columns: _,
        } = file_options;

        let source = sources.iter().next().unwrap();
        let memslice = source.to_memslice()?;
        let metadata = match metadata.take() {
            Some(md) => md,
            None => Arc::new(read_file_metadata(&mut std::io::Cursor::new(
                memslice.as_ref(),
            ))?),
        };

        // Always create a slice. If no slice was given, just make the biggest slice possible.
        let slice = match slice {
            None => (0, usize::MAX),
            Some((offset, length)) if offset < 0 => {
                let file_num_rows = get_row_count_from_blocks(
                    &mut std::io::Cursor::new(memslice.as_ref()),
                    &metadata.blocks,
                )?;
                slice_offsets(offset, length, file_num_rows as usize)
            },
            Some((offset, length)) => (offset as usize, length),
        };
        let (offset, length) = slice;
        let slice = offset..offset + length;

        let projection = with_columns
            .as_ref()
            .map(|cols| columns_to_projection(cols, &metadata.schema))
            .transpose()?;
        let projection_info = projection
            .as_ref()
            .map(|p| prepare_projection(&metadata.schema, p.clone()));

        let file_path = Some(source.to_include_path_name().into());
        let source = Source {
            file_path,
            memslice: Arc::new(memslice),
            metadata,
        };

        Ok(IpcSourceNode {
            source,

            slice,
            row_index,
            projection_info,

            rechunk,
            include_file_paths,
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

impl SourceNode for IpcSourceNode {
    fn name(&self) -> &str {
        "ipc_source"
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
        false
    }

    fn spawn_source(
        &mut self,
        num_pipelines: usize,
        mut output_recv: Receiver<SourceOutput>,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<PlSmallStr>,
    ) {
        // Split size for morsels.
        let max_morsel_size = get_max_morsel_size();
        let source_token = SourceToken::new();

        let Self {
            source,
            row_index,
            slice,
            projection_info,
            rechunk,
            include_file_paths,
        } = self;

        /// Messages sent from Walker task to Decoder tasks.
        struct BatchMessage {
            row_idx_offset: IdxSize,
            slice: Range<usize>,
            block_range: Range<usize>,
            morsel_seq_base: u64,
        }

        // Walker task -> Decoder tasks.
        let (mut batch_tx, batch_rxs) =
            distributor_channel::<BatchMessage>(num_pipelines, DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        // Decoder tasks -> Distributor task.
        let (mut decoded_rx, decoded_tx) =
            Linearizer::<Priority<Reverse<MorselSeq>, DataFrame>>::new(
                num_pipelines,
                DEFAULT_LINEARIZER_BUFFER_SIZE,
            );

        // Distributor task.
        //
        // Shuffles morsels from `n` producers amongst `n` consumers.
        //
        // If record batches in the source IPC file are large, one decoder might produce many
        // morsels at the same time. At the same time, other decoders might not produce anything.
        // Therefore, we would like to distribute the output of a single decoder task over the
        // available output pipelines.
        join_handles.push(spawn(TaskPriority::High, async move {
            // Every phase we are given a new send port.
            'phase_loop: while let Ok(phase_output) = output_recv.recv().await {
                let mut sender = phase_output.port.serial();
                let source_token = SourceToken::new();
                let wait_group = WaitGroup::default();

                while let Some(Priority(Reverse(seq), df)) = decoded_rx.get().await {
                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                    morsel.set_consume_token(wait_group.token());

                    if let Some(rc) = unrestricted_row_count.as_ref() {
                        morsel = morsel.map(|mut df| {
                            df.with_column(Column::new_scalar(
                                rc.clone(),
                                Scalar::from(df.height() as IdxSize),
                                df.height(),
                            ))
                            .unwrap();
                            df
                        });
                    }

                    if sender.send(morsel).await.is_err() {
                        return Ok(());
                    }

                    wait_group.wait().await;
                    if source_token.stop_requested() {
                        phase_output.outcome.stop();
                        continue 'phase_loop;
                    }
                }

                break;
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
                let source = source.clone();
                let rechunk = *rechunk;
                let row_index = row_index.clone();
                let projection_info = projection_info.clone();
                let include_file_paths = include_file_paths.clone();
                spawn(TaskPriority::Low, async move {
                    // Amortize allocations.
                    let mut data_scratch = Vec::new();
                    let mut message_scratch = Vec::new();

                    let schema = projection_info.as_ref().map_or(source.metadata.schema.as_ref(), |ProjectionInfo { schema, .. }| schema);
                    let pl_schema = schema
                        .iter()
                        .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
                        .collect();

                    let mut reader = FileReader::new_with_projection_info(
                        Cursor::new(source.memslice.as_ref()),
                        source.metadata.as_ref().clone(),
                        projection_info.clone(),
                        None,
                    );

                    while let Ok(m) = rx.recv().await {
                        let BatchMessage {
                            row_idx_offset,
                            slice,
                            morsel_seq_base,
                            block_range,
                        } = m;

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

                        if rechunk {
                            df.rechunk_mut();
                        }

                        if let Some(RowIndex { name, offset: _ }) = &row_index {
                            let offset = row_idx_offset + slice.start as IdxSize;
                            df = df.with_row_index(name.clone(), Some(offset))?;
                        }

                        if let Some(col) = include_file_paths.as_ref() {
                            let file_path = source.file_path.as_ref().unwrap();
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
                            let morsel_df = df.slice((i * max_morsel_size) as i64, max_morsel_size);
                            let seq = MorselSeq::new(morsel_seq_base + i as u64);
                            if send.insert(Priority(Reverse(seq), morsel_df)).await.is_err() {
                                break;
                            }
                        }

                        (data_scratch, message_scratch) = reader.take_scratches();
                    }

                    PolarsResult::Ok(())
                })
            })
            .collect::<Vec<_>>();

        let source = source.clone();
        let slice = slice.clone();
        let row_index = row_index.clone();
        let projection_info = projection_info.clone();

        // Walker task.
        //
        // Walks all the sources and supplies block ranges to the decoder tasks.
        join_handles.push(spawn(TaskPriority::Low, async move {
            let mut morsel_seq: u64 = 0;
            let mut row_idx_offset: IdxSize = row_index.as_ref().map_or(0, |ri| ri.offset);
            let mut slice: Range<usize> = slice;

            struct Batch {
                row_idx_offset: IdxSize,
                block_start: usize,
                num_rows: usize,
            }

            // Batch completion parameters
            let batch_size_limit = get_ideal_morsel_size();
            let sliced_batch_size_limit = slice.len().div_ceil(num_pipelines);
            let batch_block_limit = source.metadata.blocks.len().div_ceil(num_pipelines);

            let mut reader = FileReader::new_with_projection_info(
                Cursor::new(source.memslice.as_ref()),
                source.metadata.as_ref().clone(),
                projection_info.clone(),
                None,
            );

            if slice.start > 0 {
                // Skip over all blocks that the slice would skip anyway.
                let new_offset = reader.skip_blocks_till_limit(slice.start as u64)?;

                row_idx_offset += (slice.start as u64 - new_offset) as IdxSize;
                slice = new_offset as usize..new_offset as usize + slice.len();
            }

            'read: {
                // If we skip the entire file. Don't even try to read from it.
                if reader.get_current_block() == reader.metadata().blocks.len() {
                    break 'read;
                }

                let mut batch = Batch {
                    row_idx_offset,
                    block_start: reader.get_current_block(),
                    num_rows: 0,
                };

                // We don't yet want to commit these values to the state in case this batch gets
                // cancelled.
                let mut uncommitted_slice = slice.clone();
                let mut uncommitted_row_idx_offset = row_idx_offset;
                while !slice.is_empty() {
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
                            row_idx_offset: batch.row_idx_offset,
                            slice: batch_slice,
                            morsel_seq_base: morsel_seq,
                            block_range,
                        };

                        if source_token.stop_requested() {
                            break 'read;
                        }

                        if batch_tx.send(message).await.is_err() {
                            // This should only happen if the receiver of the decoder
                            // has broken off, meaning no further input will be needed.
                            break 'read;
                        }

                        // Commit the changes to the state.
                        // Now, we know that the a decoder will process it.
                        //
                        // This might generate several morsels if the record batch is very large.
                        morsel_seq += batch_slice_len.div_ceil(max_morsel_size) as u64;
                        slice = uncommitted_slice.clone();
                        row_idx_offset = uncommitted_row_idx_offset;

                        batch = Batch {
                            row_idx_offset,
                            block_start: current_block,
                            num_rows: 0,
                        };
                    }
                }
            } // 'read

            drop(batch_tx); // Inform decoder tasks to stop.
            for decoder_task in decoder_tasks {
                decoder_task.await?;
            }

            PolarsResult::Ok(())
        }));
    }
}

impl MultiScanable for IpcSourceNode {
    const BASE_NAME: &'static str = "ipc";

    const DOES_PRED_PD: bool = false;
    const DOES_SLICE_PD: bool = true;
    const DOES_ROW_INDEX: bool = true;

    async fn new(source: ScanSource) -> PolarsResult<Self> {
        let source = source.into_sources();
        let options = IpcScanOptions;

        let memslice = source.at(0).to_memslice()?;
        let metadata = Arc::new(read_file_metadata(&mut std::io::Cursor::new(
            memslice.as_ref(),
        ))?);

        let arrow_schema = metadata.schema.clone();
        let schema = Schema::from_arrow_schema(arrow_schema.as_ref());

        let file_options = FileScanOptions::default();
        let file_info = FileInfo::new(
            Arc::new(schema),
            Some(rayon::iter::Either::Left(arrow_schema)),
            (None, usize::MAX),
        );

        IpcSourceNode::new(source, file_info, options, None, file_options, None)
    }

    fn with_projection(&mut self, projection: Option<&Bitmap>) {
        self.projection_info = projection.map(|p| {
            let p = p.true_idx_iter().collect();
            prepare_projection(&self.source.metadata.schema, p)
        });
    }
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestrication>) {
        self.slice = 0..usize::MAX;
        if let Some(row_restriction) = row_restriction {
            match row_restriction {
                RowRestrication::Slice(slice) => self.slice = slice,
                RowRestrication::Predicate(_) => unreachable!(),
            }
        }
    }
    fn with_row_index(&mut self, row_index: Option<PlSmallStr>) {
        self.row_index = row_index.map(|name| RowIndex { name, offset: 0 });
    }

    async fn row_count(&mut self) -> PolarsResult<IdxSize> {
        get_row_count_from_blocks(
            &mut std::io::Cursor::new(self.source.memslice.as_ref()),
            &self.source.metadata.blocks,
        )
        .map(|v| v as IdxSize)
    }
    async fn schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(Arc::new(Schema::from_arrow_schema(
            &self.source.metadata.schema,
        )))
    }
}
