use std::io::Cursor;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, get_row_count_from_blocks,
};
use polars_error::{PolarsResult, polars_err};
use polars_io::utils::byte_source::{BufferByteSource, ByteSource, DynByteSource};
use polars_utils::IdxSize;
use polars_utils::relaxed_cell::RelaxedCell;
use tokio::sync::mpsc::Sender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::async_primitives::oneshot_channel;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::metrics::OptIOMetrics;
use crate::nodes::io_sources::ipc::ROW_COUNT_OVERFLOW_ERR;
use crate::utils::tokio_handle_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RecordBatchData {
    pub(super) fetched_bytes: Buffer<u8>,
    pub(super) block_index: usize,
    pub(super) num_rows: usize,
    // Lazily updated.
    pub(super) row_offset: Option<IdxSize>,
}

pub(super) struct RecordBatchDataFetcher {
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) record_batch_idx: usize,
    pub(super) fetch_metadata_only: bool,
    pub(super) n_rows_limit: Option<usize>,
    pub(super) n_rows_in_file_tx: Option<oneshot_channel::Sender<IdxSize>>,
    pub(super) row_position_on_end_tx: Option<oneshot_channel::Sender<IdxSize>>,
    pub(super) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>,
        OwnedSemaphorePermit,
    )>,
    pub(super) rb_prefetch_semaphore: Arc<Semaphore>,
    pub(super) rb_prefetch_current_all_spawned: Option<WaitToken>,
    pub(super) io_metrics: OptIOMetrics,
}

impl RecordBatchDataFetcher {
    pub(super) async fn run(&mut self) -> PolarsResult<()> {
        let current_row_offset: Arc<RelaxedCell<u64>> = Arc::new(RelaxedCell::new_u64(0));
        let row_count_updated: WaitGroup = WaitGroup::default();
        let n_record_batches = self.metadata.blocks.len();

        if !self.fetch_metadata_only {
            // Fetch all record batch data until n_rows_limit.

            while self.record_batch_idx < n_record_batches {
                if let Some(n_rows_limit) = self.n_rows_limit {
                    if current_row_offset.as_ref().load() > n_rows_limit as u64 {
                        break;
                    }
                }

                let fetch_permit = self
                    .rb_prefetch_semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .unwrap();

                let block_index = self.record_batch_idx;
                let file_metadata = self.metadata.clone();
                let current_byte_source = self.byte_source.clone();
                let memory_prefetch_func = self.memory_prefetch_func;
                let io_runtime = polars_io::pl_async::get_runtime();
                let io_metrics = self.io_metrics.clone();

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
                            let fut = current_byte_source.get_range(range.clone());
                            io_metrics.record_download(range.len() as u64, fut).await?
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
                    .send((handle, fetch_permit))
                    .await
                    .is_err()
                {
                    break;
                }

                self.record_batch_idx += 1;
            }
        };

        // Remaining row count task.
        let rb_prefetch_current_all_spawned = self.rb_prefetch_current_all_spawned.take();
        let remaining_rows_fetch_task =
            if self.n_rows_in_file_tx.is_some() && self.record_batch_idx < n_record_batches {
                let io_runtime = polars_io::pl_async::get_runtime();

                let byte_source = self.byte_source.clone();
                let io_metrics = self.io_metrics.clone();
                let file_metadata = self.metadata.clone();
                let current_idx = self.record_batch_idx;

                Some(io_runtime.spawn(async move {
                    let out = Self::fetch_row_count(
                        byte_source,
                        io_metrics,
                        file_metadata,
                        Some(current_idx),
                    )
                    .await;
                    drop(rb_prefetch_current_all_spawned);
                    out
                }))
            } else {
                drop(rb_prefetch_current_all_spawned);
                None
            };

        // Handle callback for rows fetched until the 'end', i.e. when the requested
        // slice request has been fulfilled, or somewhat higher due to latency.
        if let Some(row_position_on_end_tx) = self.row_position_on_end_tx.take() {
            row_count_updated.wait().await;

            let current_row_offset = IdxSize::try_from(current_row_offset.as_ref().load())
                .map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
            _ = row_position_on_end_tx.send(current_row_offset);
        }

        // Handle callback for total rows in file.
        // Fetch record batch metadata only.
        if let Some(n_rows_in_file_tx) = self.n_rows_in_file_tx.take() {
            row_count_updated.wait().await;

            let current_row_offset = IdxSize::try_from(current_row_offset.as_ref().load())
                .map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;

            let remaining_rows = if let Some(handle) = remaining_rows_fetch_task {
                handle.await.unwrap()?
            } else {
                0
            };
            let n_rows = current_row_offset
                .checked_add(remaining_rows)
                .ok_or(ROW_COUNT_OVERFLOW_ERR)?;
            _ = n_rows_in_file_tx.send(n_rows);
        }

        Ok(())
    }

    /// Total row count for all record batches starting at `start_offset`
    async fn fetch_row_count(
        byte_source: Arc<DynByteSource>,
        io_metrics: OptIOMetrics,
        file_metadata: Arc<FileMetadata>,
        start_offset: Option<usize>,
    ) -> PolarsResult<IdxSize> {
        let start_offset = start_offset.unwrap_or_default();

        let n_rows = match &*byte_source {
            DynByteSource::Buffer(BufferByteSource(memslice)) => {
                let n_rows: i64 = get_row_count_from_blocks(
                    &mut std::io::Cursor::new(memslice.as_ref()),
                    &file_metadata.blocks[start_offset..],
                )?;

                IdxSize::try_from(n_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?
            },
            DynByteSource::Cloud(_) => {
                let io_runtime = polars_io::pl_async::get_runtime();

                let mut n_rows = 0;
                let mut message_scratch = Vec::new();
                let mut total_bytes: u64 = 0;
                let mut ranges: Vec<_> = file_metadata.blocks[start_offset..]
                    .iter()
                    .map(|block| {
                        block.offset as usize
                            ..block.offset as usize + block.meta_data_length as usize
                    })
                    .inspect(|range| total_bytes += range.len() as u64)
                    .collect();
                let ranges_len = ranges.len();

                let bytes_map = io_runtime
                    .spawn(async move {
                        let fut = byte_source.get_ranges(&mut ranges);
                        match byte_source.as_ref() {
                            DynByteSource::Buffer(_) => fut.await,
                            DynByteSource::Cloud(_) => {
                                io_metrics.record_download(total_bytes, fut).await
                            },
                        }
                    })
                    .await
                    .unwrap()?;
                assert_eq!(bytes_map.len(), ranges_len);

                for bytes in bytes_map.into_values() {
                    let mut reader = BlockReader::new(Cursor::new(bytes.as_ref()));
                    n_rows += reader.record_batch_num_rows(&mut message_scratch)?;
                }

                IdxSize::try_from(n_rows)
                    .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?
            },
        };

        Ok(n_rows)
    }
}
