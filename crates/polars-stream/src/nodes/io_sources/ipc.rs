use std::cmp::Reverse;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::TryExtend;
use async_trait::async_trait;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::utils::arrow::io::ipc::read::{
    FileMetadata, ProjectionInfo, get_row_count_from_blocks, prepare_projection, read_file_metadata,
};
use polars_error::{ErrString, PolarsError, PolarsResult, polars_err};
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::{ScanSource, ScanSourceRef};
use polars_utils::IdxSize;
use polars_utils::mmap::MemSlice;
use polars_utils::priority::Priority;
use polars_utils::slice_enum::Slice;

use super::multi_file_reader::reader_interface::output::FileReaderOutputRecv;
use super::multi_file_reader::reader_interface::{BeginReadArgs, calc_row_position_after_slice};
use crate::async_executor::{AbortOnDropHandle, JoinHandle, TaskPriority, spawn};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    FileReader, FileReaderCallbacks,
};
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub mod builder {
    use std::sync::Arc;

    use arrow::io::ipc::read::FileMetadata;
    use polars_core::config;
    use polars_io::cloud::CloudOptions;
    use polars_plan::dsl::ScanSource;

    use super::IpcFileReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
    use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;

    #[derive(Debug)]
    pub struct IpcReaderBuilder {
        #[expect(unused)]
        pub first_metadata: Option<Arc<FileMetadata>>,
    }

    #[cfg(feature = "ipc")]
    impl FileReaderBuilder for IpcReaderBuilder {
        fn reader_name(&self) -> &str {
            "ipc"
        }

        fn reader_capabilities(&self) -> ReaderCapabilities {
            use ReaderCapabilities as RC;

            RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE
        }

        fn build_file_reader(
            &self,
            source: ScanSource,
            cloud_options: Option<Arc<CloudOptions>>,
            #[expect(unused)] scan_source_idx: usize,
        ) -> Box<dyn FileReader> {
            let scan_source = source;
            let verbose = config::verbose();

            // FIXME: For some reason the metadata does not match on idx == 0, and we end up with
            // * ComputeError: out-of-spec: InvalidBuffersLength { buffers_size: 1508, file_size: 763 }
            //
            // let metadata: Option<Arc<FileMetadata>> = if scan_source_idx == 0 {
            //     self.first_metadata.clone()
            // } else {
            //     None
            // };
            let metadata = None;

            let reader = IpcFileReader {
                scan_source,
                cloud_options,
                metadata,
                verbose,
                init_data: None,
            };

            Box::new(reader) as Box<dyn FileReader>
        }
    }
}

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (polars-u64-idx package on python)",
));

struct IpcFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    metadata: Option<Arc<FileMetadata>>,
    verbose: bool,

    init_data: Option<InitializedState>,
}

#[derive(Clone)]
struct InitializedState {
    memslice: MemSlice,
    file_metadata: Arc<FileMetadata>,
    // Lazily initialized - getting this involves iterating record batches.
    n_rows_in_file: Option<IdxSize>,
}

/// Move `slice` forward by `n` and return the slice until then.
fn slice_take(slice: &mut Range<usize>, n: usize) -> Range<usize> {
    let offset = slice.start;
    let length = slice.len();

    assert!(offset <= n);

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

#[async_trait]
impl FileReader for IpcFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        if self.init_data.is_some() {
            return Ok(());
        }

        // check_latest: IR resolution does not download IPC.
        // TODO: Streaming reads
        if let ScanSourceRef::Path(p) = self.scan_source.as_scan_source_ref() {
            polars_io::file_cache::init_entries_from_uri_list(
                &[Arc::from(p.to_str().unwrap())],
                self.cloud_options.as_deref(),
            )?;
        }

        let memslice = self
            .scan_source
            .as_scan_source_ref()
            .to_memslice_async_check_latest(self.scan_source.run_async())?;

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            Arc::new(read_file_metadata(&mut std::io::Cursor::new(
                memslice.as_ref(),
            ))?)
        };

        self.init_data = Some(InitializedState {
            memslice,
            file_metadata,
            n_rows_in_file: None,
        });

        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let InitializedState {
            memslice,
            file_metadata,
            n_rows_in_file: _,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projected_schema,
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

        let normalized_pre_slice = if let Some(pre_slice) = pre_slice_arg.clone() {
            Some(pre_slice.restrict_to_bounds(usize::try_from(self._n_rows_in_file()?).unwrap()))
        } else {
            None
        };

        if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
            _ = n_rows_in_file_tx.try_send(self._n_rows_in_file()?);
        }

        if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
            _ = row_position_on_end_tx
                .try_send(self._row_position_after_slice(normalized_pre_slice.clone())?);
        }

        if let Some(mut file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.try_send(file_schema_pl.clone());
        }

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();

            if verbose {
                eprintln!(
                    "[IpcFileReader]: early return: \
                    n_rows_in_file: {} \
                    pre_slice: {:?} \
                    resolved_pre_slice: {:?} \
                    ",
                    self._n_rows_in_file()?,
                    pre_slice_arg,
                    normalized_pre_slice
                )
            }

            return Ok((rx, spawn(TaskPriority::Low, std::future::ready(Ok(())))));
        }

        // Prepare parameters for tasks

        // Always create a slice. If no slice was given, just make the biggest slice possible.
        let slice: Range<usize> = normalized_pre_slice
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

        // Split size for morsels.
        let max_morsel_size = get_max_morsel_size();

        let metadata = file_metadata;

        /// Messages sent from Walker task to Decoder tasks.
        struct BatchMessage {
            row_idx_offset: IdxSize,
            slice: Range<usize>,
            block_range: Range<usize>,
            morsel_seq_base: u64,
        }

        let (mut morsel_sender, morsel_rx) = FileReaderOutputSend::new_serial();

        // Walker task -> Decoder tasks.
        let (mut batch_tx, batch_rxs) =
            distributor_channel::<BatchMessage>(num_pipelines, *DEFAULT_DISTRIBUTOR_BUFFER_SIZE);
        // Decoder tasks -> Distributor task.
        let (mut decoded_rx, decoded_tx) =
            Linearizer::<Priority<Reverse<MorselSeq>, DataFrame>>::new(
                num_pipelines,
                *DEFAULT_LINEARIZER_BUFFER_SIZE,
            );

        // Explicitly linearize here to redistribute morsels from large record batches.
        //
        // If record batches in the source IPC file are large, one decoder might produce many
        // morsels at the same time. At the same time, other decoders might not produce anything.
        // Therefore, we would like to distribute the output of a single decoder task over the
        // available output pipelines.
        //
        // Note, we can theoretically use `FileReaderOutputSend::parallel()` as it also linearizes
        // internally, but this behavior is an implementation detail rather than a guarantee.
        let distributor_handle = AbortOnDropHandle::new(spawn(TaskPriority::High, async move {
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            while let Some(Priority(Reverse(seq), df)) = decoded_rx.get().await {
                let morsel = Morsel::new(df, seq, source_token.clone());

                if morsel_sender.send_morsel(morsel).await.is_err() {
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
        let decoder_handles = decoded_tx
            .into_iter()
            .zip(batch_rxs)
            .map(|(mut send, mut rx)| {
                let memslice = memslice.clone();
                let metadata = metadata.clone();
                let row_index = row_index.clone();
                let projection_info = projection_info.clone();
                AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
                    // Amortize allocations.
                    let mut data_scratch = Vec::new();
                    let mut message_scratch = Vec::new();

                    let schema = projection_info.as_ref().map_or(
                        metadata.schema.as_ref(),
                        |ProjectionInfo { schema, .. }| schema,
                    );
                    let pl_schema = schema
                        .iter()
                        .map(|(n, f)| (n.clone(), DataType::from_arrow_field(f)))
                        .collect::<Schema>();

                    while let Ok(m) = rx.recv().await {
                        let BatchMessage {
                            row_idx_offset,
                            slice,
                            morsel_seq_base,
                            block_range,
                        } = m;

                        // If we don't project any columns we cannot read properly from the file,
                        // so we just create an empty frame with the proper height.
                        let mut df = if pl_schema.is_empty() {
                            DataFrame::empty_with_height(slice.len())
                        } else {
                            use polars_core::utils::arrow::io::ipc;

                            let mut reader = ipc::read::FileReader::new_with_projection_info(
                                Cursor::new(memslice.as_ref()),
                                metadata.as_ref().clone(),
                                projection_info.clone(),
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

                            (data_scratch, message_scratch) = reader.take_scratches();
                            df = df.slice(slice.start as i64, slice.len());

                            df
                        };

                        if let Some(RowIndex { name, offset: _ }) = &row_index {
                            let offset = row_idx_offset + slice.start as IdxSize;
                            df = df.with_row_index(name.clone(), Some(offset))?;
                        }

                        // If the block is very large, we want to split the block amongst the
                        // pipelines. That will at least allow some parallelism.
                        if df.height() > max_morsel_size && verbose {
                            eprintln!(
                                "IpcFileReader encountered a (too) large record batch \
                                of {} rows. Splitting and continuing.",
                                df.height()
                            );
                        }

                        for i in 0..df.height().div_ceil(max_morsel_size) {
                            let morsel_df = df.slice((i * max_morsel_size) as i64, max_morsel_size);
                            let seq = MorselSeq::new(morsel_seq_base + i as u64);
                            if send
                                .insert(Priority(Reverse(seq), morsel_df))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }

                    PolarsResult::Ok(())
                }))
            })
            .collect::<Vec<_>>();

        let memslice = memslice.clone();
        let metadata = metadata.clone();
        let slice = slice.clone();
        let row_index = row_index.clone();
        let projection_info = projection_info.clone();

        // Walker task.
        //
        // Walks all the sources and supplies block ranges to the decoder tasks.
        let walker_handle = AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
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
            let batch_block_limit = metadata.blocks.len().div_ceil(num_pipelines);

            use polars_core::utils::arrow::io::ipc;

            let mut reader = ipc::read::FileReader::new_with_projection_info(
                Cursor::new(memslice.as_ref()),
                metadata.as_ref().clone(),
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

            PolarsResult::Ok(())
        }));

        Ok((
            morsel_rx,
            spawn(TaskPriority::Low, async move {
                distributor_handle.await?;

                for handle in decoder_handles {
                    handle.await?;
                }

                walker_handle.await?;
                Ok(())
            }),
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
    fn _n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        let InitializedState {
            memslice,
            file_metadata,
            n_rows_in_file,
        } = self.init_data.as_mut().unwrap();

        if n_rows_in_file.is_none() {
            let n_rows: i64 = get_row_count_from_blocks(
                &mut std::io::Cursor::new(memslice.as_ref()),
                &file_metadata.blocks,
            )?;

            let n_rows = IdxSize::try_from(n_rows)
                .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

            *n_rows_in_file = Some(n_rows);
        }

        Ok(n_rows_in_file.unwrap())
    }

    fn _row_position_after_slice(&mut self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        Ok(calc_row_position_after_slice(
            self._n_rows_in_file()?,
            pre_slice,
        ))
    }
}
