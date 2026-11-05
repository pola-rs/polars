use std::io::Cursor;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_config::config;
use polars_core::runtime::ASYNC;
use polars_core::utils::polars_arrow::io::ipc::read::{BlockReader, FileMetadata};
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{PolarsResult, polars_ensure, polars_err};
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
    /// Rows within the pre-slice, relative to the record batch. `None` for all rows.
    pub(super) slice: Option<(usize, usize)>,
}

/// Largest gap between record batches that are fetched together.
const MAX_GROUP_GAP: usize = 4096;

/// A group takes at most 1 / `GROUP_BUDGET_SHARE` of the pipeline budget, so it can always be
/// filled.
const GROUP_BUDGET_SHARE: usize = 4;

/// Default largest span of a group. Larger requests leave too few in flight under the current
/// in-flight budget.
const DEFAULT_GROUP_MAX_BYTES: usize = 2 << 20;

/// A record batch waiting for its group to be fetched.
struct PendingRecordBatch {
    record_batch_idx: usize,
    num_rows: Option<IdxSize>,
    row_offset: Option<IdxSize>,
    slice: Option<(usize, usize)>,
    range: Range<usize>,
    fetch_permit: Option<PipelinePermit>,
}

/// Where the record batches of a group get their bytes from.
#[derive(Clone)]
enum GroupBytes<F> {
    /// Memory-mapped or in-memory file, sliced per record batch.
    Memory(Buffer<u8>),
    /// One `get_ranges` call for the group, shared by its record batches and dropped, cancelling
    /// the fetch, with the last of them.
    Fetched(F),
    /// No record batch of the group needs bytes.
    None,
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
        Option<PipelinePermit>,
    )>,
    pub(super) base_rb_metadata_fetch_count: u64,

    pub(super) pipeline_budget: PipelineBudget,
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

            pipeline_budget,
            rb_prefetch_current_all_spawned,
        } = self;

        let global_slice = pre_slice.clone().map(Range::<usize>::from);

        polars_ensure!(
            global_slice.is_none() || record_batch_cum_len.is_some(),
            ComputeError: "IPC pre-slice requires record batch lengths"
        );
        let mut rb_fetch_count: u64 = 0;

        // Consecutive record batches are fetched together, like Parquet row groups. Groups depend
        // only on the file and these limits, not on load.
        // Only remote sources have a chunk size, so only they group.
        // TODO: Revisit the group size once the in-flight budget is sized for large requests.
        let max_group_span = byte_source.chunk_size().map_or(0, |chunk_size| {
            let default = chunk_size.min(DEFAULT_GROUP_MAX_BYTES);
            std::env::var("POLARS_RECORD_BATCH_GROUP_MAX_BYTES").map_or(default, |x| {
                x.parse::<NonZeroUsize>()
                    .unwrap_or_else(|_| {
                        panic!("invalid value for POLARS_RECORD_BATCH_GROUP_MAX_BYTES: {x}")
                    })
                    .get()
                    .min(chunk_size)
            })
        });
        let max_group_len = (pipeline_budget.count_limit() / GROUP_BUDGET_SHARE).max(1);
        let max_group_kbytes = pipeline_budget.kbytes_limit() / GROUP_BUDGET_SHARE;
        let mut group: Vec<PendingRecordBatch> = Vec::new();
        let mut group_kbytes: usize = 0;
        let mut group_fetch_count: u64 = 0;

        for record_batch_idx in 0..file_metadata.blocks.len() {
            let mut num_rows_this_rb: Option<IdxSize> = None;
            let mut row_offset: Option<IdxSize> = None;
            let mut slice: Option<(usize, usize)> = None;

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
                        SplitSlicePosition::Overlapping(offset, len) => slice = Some((offset, len)),
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

            let permit_kbytes = pipeline_budget.permit_kbytes(fetch_length);

            // Start a new group at a gap or out-of-order block, so `get_ranges` does not fetch the
            // bytes in between, when it would span more than one request, or when it would take
            // more than its share of the budget.
            let starts_new_group = group
                .first()
                .zip(group.last())
                .is_some_and(|(first, last)| {
                    range
                        .start
                        .checked_sub(last.range.end)
                        .is_none_or(|gap| gap > MAX_GROUP_GAP)
                        || range.end - first.range.start > max_group_span
                        || group.len() >= max_group_len
                        || group_kbytes + permit_kbytes > max_group_kbytes
                });

            if starts_new_group {
                group_kbytes = 0;
                let Some(n) = spawn_group(
                    std::mem::take(&mut group),
                    &byte_source,
                    memory_prefetch_func,
                    &prefetch_send,
                )
                .await
                else {
                    break;
                };
                group_fetch_count += n;
            }

            // Load delays a group but never splits it: it takes at most a share of the budget, and
            // all other permits are held by earlier record batches, which are always consumed.
            let fetch_permit = if rb_fetch == RbFetch::None {
                None
            } else {
                Some(pipeline_budget.acquire(fetch_length).await)
            };

            group_kbytes += permit_kbytes;
            group.push(PendingRecordBatch {
                record_batch_idx,
                num_rows: num_rows_this_rb,
                row_offset,
                slice,
                range,
                fetch_permit,
            });
        }

        if let Some(n) =
            spawn_group(group, &byte_source, memory_prefetch_func, &prefetch_send).await
        {
            group_fetch_count += n;
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
                    rb_metadata_fetch_count: {rb_metadata_fetch_count}, \
                    group_fetch_count: {group_fetch_count}"
            )
        }

        Ok(())
    }
}

/// Spawns the fetch of a group of record batches and sends a handle per record batch in order.
/// Returns the number of `get_ranges` calls, or `None` if the receiver is gone.
async fn spawn_group(
    group: Vec<PendingRecordBatch>,
    byte_source: &Arc<DynByteSource>,
    memory_prefetch_func: fn(&[u8]) -> (),
    prefetch_send: &Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>,
        Option<PipelinePermit>,
    )>,
) -> Option<u64> {
    let group_bytes = match byte_source.as_ref() {
        DynByteSource::Buffer(mem_slice) => GroupBytes::Memory(mem_slice.0.clone()),
        _ => {
            let mut ranges: Vec<Range<usize>> = group
                .iter()
                .map(|rb| rb.range.clone())
                .filter(|range| !range.is_empty())
                .collect();
            if ranges.is_empty() {
                GroupBytes::None
            } else {
                let byte_source = byte_source.clone();
                // In its own task, so body chunks wake only the fetch, not every record batch task.
                let fetch =
                    tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn(async move {
                        byte_source.get_ranges(&mut ranges).await.map(Arc::new)
                    }));
                GroupBytes::Fetched(
                    fetch
                        .map(|r| {
                            r.unwrap_or_else(|e| {
                                if e.is_panic() {
                                    std::panic::resume_unwind(e.into_panic())
                                }
                                Err(polars_err!(ComputeError: "IPC record batch fetch was cancelled"))
                            })
                        })
                        .shared(),
                )
            }
        },
    };
    let n_fetches = matches!(group_bytes, GroupBytes::Fetched(_)) as u64;

    for PendingRecordBatch {
        record_batch_idx,
        num_rows,
        row_offset,
        slice,
        range,
        fetch_permit,
    } in group
    {
        let group_bytes = group_bytes.clone();

        let fetch_handle = ASYNC.spawn(async move {
            let fetched_bytes = match group_bytes {
                GroupBytes::None => Buffer::new(),
                _ if range.is_empty() => Buffer::new(),
                GroupBytes::Memory(mem) => {
                    if !std::ptr::eq(
                        memory_prefetch_func as *const (),
                        polars_utils::mem::prefetch::no_prefetch as *const (),
                    ) {
                        debug_assert!(range.end <= mem.len());
                        memory_prefetch_func(unsafe { mem.as_ref().get_unchecked(range.clone()) })
                    }

                    mem.sliced(range)
                },
                GroupBytes::Fetched(group_fetch) => group_fetch
                    .await?
                    .get(&range.start)
                    .filter(|bytes| bytes.len() == range.len())
                    .cloned()
                    .ok_or_else(|| {
                        polars_err!(ComputeError: "IPC record batch {record_batch_idx} not in its group fetch")
                    })?,
            };

            // Extract the length (i.e., nr of rows) at the earliest possible opportunity.
            let num_rows = if let Some(num_rows) = num_rows {
                num_rows
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
                slice,
            })
        });

        let fetch_handle = tokio_handle_ext::AbortOnDropHandle(fetch_handle);

        if prefetch_send
            .send((fetch_handle, fetch_permit))
            .await
            .is_err()
        {
            return None;
        }
    }

    Some(n_fetches)
}
