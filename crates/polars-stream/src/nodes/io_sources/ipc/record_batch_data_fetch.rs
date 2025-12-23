use std::io::Cursor;
use std::sync::Arc;

use polars_core::utils::arrow::io::ipc::read::{BlockReader, FileMetadata, ProjectionInfo};
use polars_error::PolarsResult;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_utils::mmap::MemSlice;

use crate::utils::task_handles_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
// kdn TODO clean up fields
pub(super) struct RecordBatchData {
    pub(super) fetched_bytes: MemSlice,
    // pub(super) row_offset: usize,
    // pub(super) slice: Option<(usize, usize)>,
    // pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) file_metadata: Arc<FileMetadata>,
    // pub(super) sorting_map: Vec<(usize, IsSorted)>,
    pub(super) num_rows: usize,
}

// kdn TODO clean up fields
pub(super) struct RecordBatchDataFetcher {
    pub(super) projection_info: Arc<Option<ProjectionInfo>>,
    // pub(super) projection: Arc<[ArrowFieldProjection]>,
    // pub(super) is_full_projection: bool,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) record_batch_idx: usize,
}

impl RecordBatchDataFetcher {
    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>>>
    {
        let idx = self.record_batch_idx;
        self.record_batch_idx += 1;

        let file_metadata = self.metadata.clone();

        if idx == file_metadata.blocks.len() {
            return None;
        }

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
                    let bytes = current_byte_source.get_range(range).await?;
                    bytes
                };

            // We extract the length (i.e., nr of rows) at the earliest possible time.
            let num_rows = {
                let mut reader = BlockReader::new(
                    Cursor::new(fetched_bytes.as_ref()),
                    file_metadata.as_ref(),
                    None,
                );
                let mut message_scratch = Vec::new();
                reader.record_batch_num_rows(&mut message_scratch)?
            };

            PolarsResult::Ok(RecordBatchData {
                fetched_bytes,
                file_metadata,
                num_rows,
            })
        });

        let handle = task_handles_ext::AbortOnDropHandle(handle);
        return Some(Ok(handle));
    }
}
