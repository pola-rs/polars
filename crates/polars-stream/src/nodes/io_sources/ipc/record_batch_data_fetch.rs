use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, ProjectionInfo, record_batch_num_rows,
};
use polars_error::PolarsResult;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_utils::mmap::MemSlice;

use crate::utils::task_handles_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RecordBatchData {
    pub(super) fetched_bytes: MemSlice,
    // pub(super) row_offset: usize,
    // pub(super) slice: Option<(usize, usize)>,
    // pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) file_metadata: Arc<FileMetadata>,
    // pub(super) sorting_map: Vec<(usize, IsSorted)>,
    pub(super) idx: usize,
    pub(super) num_rows: usize,
}

pub(super) struct RecordBatchDataFetcher {
    pub(super) projection_info: Arc<Option<ProjectionInfo>>,
    // pub(super) projection: Arc<[ArrowFieldProjection]>,
    // pub(super) is_full_projection: bool,
    // #[allow(unused)] // TODO: Fix!
    // pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) slice_range: Option<Range<usize>>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) record_batch_idx: usize,
    pub(super) row_idx: usize,
    // pub(super) row_group_mask: Option<Bitmap>,
    // pub(super) row_offset: usize,
}

impl RecordBatchDataFetcher {
    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>>>
    {
        // dbg!("start next for impl RBDF"); //kdn

        // kdn TODO - support slices
        debug_assert!(self.slice_range.is_none());

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
            let fetched_bytes =
                if let DynByteSource::MemSlice(mem_slice) = current_byte_source.as_ref() {
                    // Skip byte range calculation for `no_prefetch`.
                    if memory_prefetch_func as usize
                        != polars_utils::mem::prefetch::no_prefetch as usize
                    {
                        let slice = mem_slice.0.as_ref();

                        // // kdn TODO - projection
                        // if !is_full_projection {
                        //     for range in get_row_group_byte_ranges_for_projection(
                        //         row_group_metadata,
                        //         &mut projection.iter().map(|x| &x.arrow_field().name),
                        //     ) {
                        //         memory_prefetch_func(unsafe { slice.get_unchecked(range) })
                        //     }
                        // } else {
                        let block = file_metadata.blocks.get(idx).unwrap();
                        let range = block.offset as usize
                            ..block.offset as usize
                                + block.meta_data_length as usize
                                + block.body_length as usize; //kdn check i32/usize
                        memory_prefetch_func(unsafe { slice.get_unchecked(range) })
                        // };
                    }

                    // kdn TODO cleanup?
                    // // We have a mmapped or in-memory slice representing the entire
                    // // file that can be sliced directly, so we can skip the byte-range
                    // // calculations and HashMap allocation.
                    // let mem_slice = mem_slice.0.clone();
                    // mem_slice
                    // let slice = mem_slice.0.as_ref();
                    let block = file_metadata.blocks.get(idx).unwrap();
                    let range = block.offset as usize
                        ..block.offset as usize
                            + block.meta_data_length as usize
                            + block.body_length as usize; //kdn check i32/usize
                    mem_slice.0.slice(range)
                } else {
                    //kdn TODO: get_range() vs get_ranges()

                    //kdn TODO: record batch metadata

                    let block = file_metadata.blocks.get(idx).unwrap();
                    let range = block.offset as usize
                        ..block.offset as usize
                            + block.meta_data_length as usize
                            + block.body_length as usize; //kdn check i32/usize
                    let bytes = current_byte_source.get_range(range).await?;

                    bytes
                };

            // We extract the length (i.e., nr of rows) at the earliest possible time.
            let num_rows = {
                let mut reader = BlockReader::new(
                    Cursor::new(fetched_bytes.as_ref()),
                    file_metadata.as_ref().clone(),
                    None
                );
                let mut message_scratch = Vec::new(); //kdn TODO avoid allocation
                record_batch_num_rows(
                    &mut reader.reader,
                    &file_metadata,
                    0,
                    true,
                    &mut message_scratch,
                )?
            };

            PolarsResult::Ok(RecordBatchData {
                fetched_bytes,
                file_metadata,
                idx,
                num_rows,
            })
        });

        let handle = task_handles_ext::AbortOnDropHandle(handle);
        return Some(Ok(handle));
    }
}
