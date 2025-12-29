use std::io::Cursor;
use std::sync::Arc;

use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata};
use polars_error::PolarsResult;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_utils::mmap::MemSlice;
use tokio::sync::mpsc::Sender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::utils::tokio_handle_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RecordBatchData {
    pub(super) fetched_bytes: MemSlice,
    pub(super) num_rows: usize,
}

pub(super) struct RecordBatchDataFetcher {
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) record_batch_idx: usize,
    pub(super) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>,
        OwnedSemaphorePermit,
    )>,
    pub(super) rb_prefetch_semaphore: Arc<Semaphore>,
}

impl RecordBatchDataFetcher {
    pub(super) async fn run(&mut self) -> PolarsResult<()> {
        while self.record_batch_idx < self.metadata.blocks.len() {
            let fetch_permit = self
                .rb_prefetch_semaphore
                .clone()
                .acquire_owned()
                .await
                .unwrap();

            let idx = self.record_batch_idx;
            let file_metadata = self.metadata.clone();
            let current_byte_source = self.byte_source.clone();
            let memory_prefetch_func = self.memory_prefetch_func;
            let io_runtime = polars_io::pl_async::get_runtime();

            // Create future and get handle
            let handle = io_runtime.spawn(async move {
                let block = file_metadata.blocks.get(idx).unwrap();
                let range = block.offset as usize
                    ..block.offset as usize
                        + block.meta_data_length as usize
                        + block.body_length as usize;

                let fetched_bytes =
                    if let DynByteSource::MemSlice(mem_slice) = current_byte_source.as_ref() {
                        let slice = mem_slice.0.as_ref();

                        if memory_prefetch_func as usize
                            != polars_utils::mem::prefetch::no_prefetch as usize
                        {
                            debug_assert!(range.end <= slice.len());
                            memory_prefetch_func(unsafe { slice.get_unchecked(range.clone()) })
                        }

                        mem_slice.0.slice(range)
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

                PolarsResult::Ok(RecordBatchData {
                    fetched_bytes,
                    num_rows,
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

        Ok(())
    }
}
