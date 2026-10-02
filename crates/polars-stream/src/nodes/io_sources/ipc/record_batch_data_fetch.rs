use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_config::config;
use polars_core::runtime::ASYNC;
use polars_core::utils::polars_arrow::io::ipc::read::{BlockReader, FileMetadata};
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{PolarsResult, polars_err};
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;
use tokio::sync::mpsc::Sender;

use super::super::shared::pipeline_budget::{PipelineBudget, PipelinePermit};
use crate::utils::tokio_handle_ext;

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

    pub(super) prefetch_send: Sender<RecordBatchFetchHandle>,
    pub(super) base_rb_metadata_fetch_count: u64,

    pub(super) pipeline_budget: PipelineBudget,
    pub(super) rb_prefetch_current_all_spawned: Option<WaitToken>,
}

/// A spawned fetch of one or more record batches, with the budget permit for its bytes.
pub(super) type RecordBatchFetchHandle = (
    tokio_handle_ext::AbortOnDropHandle<PolarsResult<Vec<RecordBatchData>>>,
    Option<PipelinePermit>,
);

/// The byte range of one record batch to fetch.
struct RecordBatchFetch {
    record_batch_idx: usize,
    num_rows: Option<IdxSize>,
    row_offset: Option<IdxSize>,
    range: Range<usize>,
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

            pipeline_budget,
            rb_prefetch_current_all_spawned,
        } = self;

        let global_slice = pre_slice.clone().map(Range::<usize>::from);
        let mut rb_fetch_count: u64 = 0;

        // Fetch consecutive record batches from the cloud together, as the Parquet reader does
        // per row group, so `get_ranges` can coalesce them into fewer requests.
        let group_byte_limit = match byte_source.as_ref() {
            DynByteSource::Buffer(_) | DynByteSource::File(_) => 0,
            _ => FetchConfig::random_access().chunk_size,
        };
        let mut group: Vec<RecordBatchFetch> = Vec::new();
        let mut group_bytes = 0;

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

            let block = file_metadata.blocks.get(record_batch_idx).unwrap();
            let fetch_length = match rb_fetch {
                RbFetch::None => 0,
                RbFetch::Metadata => block.meta_data_length as usize,
                RbFetch::All => block.meta_data_length as usize + block.body_length as usize,
            };
            let range = block.offset as usize
                ..usize::checked_add(block.offset as _, fetch_length)
                    .ok_or_else(|| polars_err!(ComputeError: "IPC block range overflows usize"))?;

            if !group.is_empty() && group_bytes + range.len() > group_byte_limit {
                let group = std::mem::take(&mut group);
                let group_bytes = std::mem::take(&mut group_bytes);
                if !spawn_fetch(
                    group,
                    group_bytes,
                    &byte_source,
                    memory_prefetch_func,
                    &pipeline_budget,
                    &prefetch_send,
                )
                .await
                {
                    break;
                }
            }

            group_bytes += range.len();
            group.push(RecordBatchFetch {
                record_batch_idx,
                num_rows: num_rows_this_rb,
                row_offset,
                range,
            });
        }

        if !group.is_empty() {
            spawn_fetch(
                group,
                group_bytes,
                &byte_source,
                memory_prefetch_func,
                &pipeline_budget,
                &prefetch_send,
            )
            .await;
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

/// Spawns the fetch of a group of record batches. Returns `false` if the receiver is gone.
async fn spawn_fetch(
    group: Vec<RecordBatchFetch>,
    group_bytes: usize,
    byte_source: &Arc<DynByteSource>,
    memory_prefetch_func: fn(&[u8]) -> (),
    pipeline_budget: &PipelineBudget,
    prefetch_send: &Sender<RecordBatchFetchHandle>,
) -> bool {
    let fetch_permit = if group_bytes > 0 {
        Some(pipeline_budget.acquire(group_bytes).await)
    } else {
        None
    };

    let byte_source = byte_source.clone();
    let fetch_handle = ASYNC.spawn(async move {
        fetch_record_batches(&byte_source, memory_prefetch_func, group).await
    });
    let fetch_handle = tokio_handle_ext::AbortOnDropHandle(fetch_handle);

    prefetch_send
        .send((fetch_handle, fetch_permit))
        .await
        .is_ok()
}

async fn fetch_record_batches(
    byte_source: &DynByteSource,
    memory_prefetch_func: fn(&[u8]) -> (),
    group: Vec<RecordBatchFetch>,
) -> PolarsResult<Vec<RecordBatchData>> {
    let fetched: Vec<Buffer<u8>> = if let DynByteSource::Buffer(mem_slice) = byte_source {
        let slice = mem_slice.0.as_ref();

        group
            .iter()
            .map(|fetch| {
                let range = fetch.range.clone();

                if !range.is_empty()
                    && !std::ptr::eq(
                        memory_prefetch_func as *const (),
                        polars_utils::mem::prefetch::no_prefetch as *const (),
                    )
                {
                    debug_assert!(range.end <= slice.len());
                    memory_prefetch_func(unsafe { slice.get_unchecked(range.clone()) })
                }

                mem_slice.0.clone().sliced(range)
            })
            .collect()
    } else {
        let mut ranges: Vec<Range<usize>> = group
            .iter()
            .map(|fetch| fetch.range.clone())
            .filter(|range| !range.is_empty())
            .collect();
        let mut bytes_map = if ranges.is_empty() {
            Default::default()
        } else {
            byte_source.get_ranges(&mut ranges).await?
        };

        group
            .iter()
            .map(|fetch| {
                if fetch.range.is_empty() {
                    Buffer::new()
                } else {
                    bytes_map.remove(&fetch.range.start).unwrap()
                }
            })
            .collect()
    };

    group
        .into_iter()
        .zip(fetched)
        .map(|(fetch, fetched_bytes)| {
            // Extract the length (i.e., nr of rows) at the earliest possible opportunity.
            let num_rows = if let Some(num_rows) = fetch.num_rows {
                num_rows
            } else {
                let mut reader = BlockReader::new(Cursor::new(fetched_bytes.as_ref()));
                let mut message_scratch = vec![];
                reader
                    .record_batch_num_rows(&mut message_scratch)?
                    .try_into()
                    .map_err(|_| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?
            };

            Ok(RecordBatchData {
                fetched_bytes,
                record_batch_idx: fetch.record_batch_idx,
                num_rows,
                row_offset: fetch.row_offset,
            })
        })
        .collect()
}
