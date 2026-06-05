use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_config::config;
use polars_core::runtime::ASYNC;
use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata};
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{PolarsResult, polars_err};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;
use tokio::sync::mpsc::Sender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::utils::tokio_handle_ext;

pub(crate) struct PipelinePermit {
    _count: OwnedSemaphorePermit,
    _kbytes: OwnedSemaphorePermit,
}

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RecordBatchData {
    pub(super) fetched_bytes: Buffer<u8>,
    pub(super) record_batch_idx: usize,
    pub(super) num_rows: IdxSize,
    pub(super) row_offset: Option<IdxSize>,
}

pub(super) struct RecordBatchDataFetcher {
    pub(super) file_metadata: Arc<FileMetadata>,
    pub(super) record_batch_cum_len: Option<Buffer<IdxSize>>,

    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),

    /// Column indices. Full projection if `None`.
    pub(super) subset_projection_idxs: Option<Arc<[usize]>>,
    pub(super) pre_slice: Option<Slice>,

    pub(super) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>,
        //kdn TODO CLEANUP
        Option<OwnedSemaphorePermit>,
        PipelinePermit,
    )>,
    pub(super) base_rb_metadata_fetch_count: u64,

    pub(super) rb_prefetch_semaphore: Arc<Semaphore>,
    pub(super) rb_prefetch_kbytes_semaphore: Arc<Semaphore>,
    pub(super) rb_prefetch_current_all_spawned: Option<WaitToken>,
}

impl RecordBatchDataFetcher {
    pub(super) async fn run(self) -> PolarsResult<()> {
        let Self {
            file_metadata,
            record_batch_cum_len,

            byte_source,
            memory_prefetch_func,

            subset_projection_idxs,
            pre_slice,

            prefetch_send,
            base_rb_metadata_fetch_count,

            rb_prefetch_semaphore,
            rb_prefetch_current_all_spawned,
        } = self;

        let global_slice = pre_slice.clone().map(Range::<usize>::from);
        let mut rb_fetch_count: u64 = 0;

        for record_batch_idx in 0..file_metadata.blocks.len() {
            let mut num_rows_this_rb: Option<IdxSize> = None;
            let mut row_offset: Option<IdxSize> = None;

            if let Some(record_batch_cum_len) = record_batch_cum_len.as_deref() {
                row_offset = Some(
                    record_batch_idx
                        .checked_sub(1)
                        .map_or(0, |prev_idx| record_batch_cum_len[prev_idx]),
                );
                num_rows_this_rb =
                    Some(record_batch_cum_len[record_batch_idx] - row_offset.unwrap());

                if let Some(global_slice) = global_slice.clone() {
                    match SplitSlicePosition::split_slice_at_file(
                        row_offset.unwrap() as usize,
                        num_rows_this_rb.unwrap() as usize,
                        global_slice,
                    ) {
                        SplitSlicePosition::Before => continue,
                        SplitSlicePosition::Overlapping(_, _) => {},
                        SplitSlicePosition::After => break,
                    }
                }

                let block_index = self.record_batch_idx;
                let file_metadata = self.metadata.clone();

                // Acquire permits
                let block = file_metadata.blocks.get(block_index).unwrap();
                let n_bytes = block.meta_data_length as usize + block.body_length as usize;
                let n_kbytes = n_bytes.div_ceil(1 << 10).try_into().unwrap_or(u32::MAX);
                let kbytes_permit = self
                    .rb_prefetch_kbytes_semaphore
                    .clone()
                    .acquire_many_owned(n_kbytes)
                    .await
                    .unwrap();

                let count_permit = self
                    .rb_prefetch_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();

                let permit = PipelinePermit {
                    _count: count_permit,
                    _kbytes: kbytes_permit,
                };

                // Spawn task
                let current_byte_source = self.byte_source.clone();
                let memory_prefetch_func = self.memory_prefetch_func;
                let io_runtime = polars_io::pl_async::get_runtime();

                let current_row_offset = current_row_offset.clone();
                let wait_token = row_count_updated.token();

                let handle = io_runtime.spawn(async move {
                    let block = file_metadata.blocks.get(block_index).unwrap();
                    let range = block.offset as usize
                        ..block.offset as usize
                            + block.meta_data_length as usize
                            + block.body_length as usize;

                    let fetched_bytes =
                        if let DynByteSource::Buffer(mem_slice) = current_byte_source.as_ref() {
                            let slice = mem_slice.0.as_ref();

                            if !std::ptr::eq(
                                memory_prefetch_func as *const (),
                                polars_utils::mem::prefetch::no_prefetch as *const (),
                            ) {
                                debug_assert!(range.end <= slice.len());
                                memory_prefetch_func(unsafe { slice.get_unchecked(range.clone()) })
                            }

                            mem_slice.0.clone().sliced(range)
                        } else {
                            // @NOTE. Performance can be optimized by grouping requests and downloading
                            // through `get_ranges()`.
                            current_byte_source.get_range(range).await?
                        };

                    // We extract the length (i.e., nr of rows) at the earliest possible opportunity.
                    let num_rows = {
                        let mut reader = BlockReader::new(Cursor::new(fetched_bytes.as_ref()));
                        let mut message_scratch = Vec::new();
                        reader.record_batch_num_rows(&mut message_scratch)?
                    };

                    current_row_offset.fetch_add(num_rows as u64);
                    drop(wait_token);

                    PolarsResult::Ok(RecordBatchData {
                        fetched_bytes,
                        block_index,
                        num_rows,
                        row_offset: None,
                    })
                });

                let handle = tokio_handle_ext::AbortOnDropHandle(handle);
                if self
                    .prefetch_send
                    .send((handle, permit))
                    .await
                    .is_err()
                {
                    break;
                }

                self.record_batch_idx += 1;
            }

            #[derive(Debug, PartialEq)]
            enum RbFetch {
                All,
                Metadata,
                None,
            }

            let rb_fetch = if subset_projection_idxs
                .as_ref()
                .is_some_and(|x| x.is_empty())
            {
                // 0-length projection or slice.
                if num_rows_this_rb.is_some() {
                    RbFetch::None
                } else {
                    rb_fetch_count += 1 << 32;
                    RbFetch::Metadata
                }
            } else {
                rb_fetch_count += 1;
                RbFetch::All
            };

            let file_metadata = file_metadata.clone();
            let current_byte_source = byte_source.clone();
            let fetch_permit = if rb_fetch != RbFetch::None {
                Some(rb_prefetch_semaphore.clone().acquire_owned().await.unwrap())
            } else {
                None
            };

            let fetch_handle = ASYNC.spawn(async move {
                let block = file_metadata.blocks.get(record_batch_idx).unwrap();
                let fetch_length = match rb_fetch {
                    RbFetch::None => 0,
                    RbFetch::Metadata => block.meta_data_length as usize,
                    RbFetch::All => block.meta_data_length as usize + block.body_length as usize,
                };

                let range = block.offset as usize
                    ..usize::checked_add(block.offset as _, fetch_length).unwrap();

                let fetched_bytes = if range.is_empty() {
                    Buffer::new()
                } else if let DynByteSource::Buffer(mem_slice) = current_byte_source.as_ref() {
                    let slice = mem_slice.0.as_ref();

                    if !std::ptr::eq(
                        memory_prefetch_func as *const (),
                        polars_utils::mem::prefetch::no_prefetch as *const (),
                    ) {
                        debug_assert!(range.end <= slice.len());
                        memory_prefetch_func(unsafe { slice.get_unchecked(range.clone()) })
                    }

                    mem_slice.0.clone().sliced(range)
                } else {
                    // @NOTE. Performance can be optimized by grouping requests and downloading
                    // through `get_ranges()`.
                    current_byte_source.get_range(range).await?
                };

                // We extract the length (i.e., nr of rows) at the earliest possible opportunity.
                let num_rows = if let Some(num_rows_this_rb) = num_rows_this_rb {
                    num_rows_this_rb
                } else {
                    let mut reader = BlockReader::new(Cursor::new(fetched_bytes.as_ref()));
                    let mut message_scratch = vec![];
                    reader
                        .record_batch_num_rows(&mut message_scratch)?
                        .try_into()
                        .map_err(|_| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?
                };

                PolarsResult::Ok(RecordBatchData {
                    fetched_bytes,
                    record_batch_idx,
                    num_rows,
                    row_offset,
                })
            });

            let fetch_handle = tokio_handle_ext::AbortOnDropHandle(fetch_handle);

            if prefetch_send
                .send((fetch_handle, fetch_permit))
                .await
                .is_err()
            {
                break;
            }
        }

        drop(rb_prefetch_current_all_spawned);

        if config().verbose() {
            let rb_total_count = file_metadata.blocks.len();
            let rb_full_fetch_count = rb_fetch_count & ((1 << 32) - 1);
            let rb_metadata_fetch_count = (rb_fetch_count >> 32) + base_rb_metadata_fetch_count;

            eprintln!(
                "[IpcFileReader]: RecordBatchDataFetcher: \
                rb_total_count: {rb_total_count}, \
                rb_full_fetch_count: {rb_full_fetch_count}, \
                rb_metadata_fetch_count: {rb_metadata_fetch_count}"
            )
        }

        Ok(())
    }
}
