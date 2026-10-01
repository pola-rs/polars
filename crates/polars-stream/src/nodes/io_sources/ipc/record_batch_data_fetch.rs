use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use polars_async::primitives::wait_group::WaitToken;
use polars_buffer::Buffer;
use polars_config::config;
use polars_core::runtime::ASYNC;
use polars_core::utils::polars_arrow::io::ipc::read::{BlockReader, FileMetadata};
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{PolarsResult, polars_err};
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

    pub(super) prefetch_send: Sender<(
        tokio_handle_ext::AbortOnDropHandle<PolarsResult<RecordBatchData>>,
        Option<PipelinePermit>,
    )>,
    pub(super) base_rb_metadata_fetch_count: u64,

    pub(super) pipeline_budget: PipelineBudget,
    pub(super) rb_prefetch_current_all_spawned: Option<WaitToken>,
}

/// One record batch to fetch, as planned from the file metadata.
struct PlannedFetch {
    record_batch_idx: usize,
    num_rows_this_rb: Option<IdxSize>,
    row_offset: Option<IdxSize>,
    range: Range<usize>,
}

/// EXPERIMENT: byte budget for fetching consecutive record batches with one request.
fn coalesce_bytes_limit(byte_source: &DynByteSource) -> Option<usize> {
    if matches!(
        byte_source,
        DynByteSource::Buffer(_) | DynByteSource::File(_)
    ) {
        return None;
    }
    std::env::var("POLARS_IPC_COALESCE_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

async fn record_batch_data(
    fetched_bytes: Buffer<u8>,
    record_batch_idx: usize,
    num_rows_this_rb: Option<IdxSize>,
    row_offset: Option<IdxSize>,
) -> PolarsResult<RecordBatchData> {
    // Extract the length (i.e., nr of rows) at the earliest possible opportunity.
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

    Ok(RecordBatchData {
        fetched_bytes,
        record_batch_idx,
        num_rows,
        row_offset,
    })
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
        let coalesce_limit = coalesce_bytes_limit(&byte_source);

        #[derive(Debug, PartialEq)]
        enum RbFetch {
            All,
            Metadata,
            None,
        }

        // Plan all fetches up front so consecutive full fetches can be grouped.
        let mut planned: Vec<(PlannedFetch, RbFetch)> = Vec::new();
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

            planned.push((
                PlannedFetch {
                    record_batch_idx,
                    num_rows_this_rb,
                    row_offset,
                    range,
                },
                rb_fetch,
            ));
        }

        let mut i = 0;
        'outer: while i < planned.len() {
            // Group consecutive, contiguous full fetches up to the coalesce limit.
            let mut j = i + 1;
            if let Some(limit) = coalesce_limit
                && planned[i].1 == RbFetch::All
            {
                let start = planned[i].0.range.start;
                let mut end = planned[i].0.range.end;
                while j < planned.len()
                    && planned[j].1 == RbFetch::All
                    && planned[j].0.range.start == end
                    && planned[j].0.range.end - start <= limit
                {
                    end = planned[j].0.range.end;
                    j += 1;
                }
            }

            if j - i > 1 {
                // Each record batch keeps its own budget permit, as without coalescing.
                let mut permits = Vec::with_capacity(j - i);
                for (p, _) in &planned[i..j] {
                    permits.push(pipeline_budget.acquire(p.range.len()).await);
                }

                let group_range = planned[i].0.range.start..planned[j - 1].0.range.end;
                let group_start = group_range.start;
                let current_byte_source = byte_source.clone();
                let group_fetch = tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn(async move {
                    current_byte_source
                        .get_range(group_range)
                        .await
                        .map_err(Arc::new)
                }));
                let group_fetch = async move {
                    match group_fetch.await {
                        Ok(v) => v,
                        Err(e) => Err(Arc::new(polars_err!(ComputeError: "{e}"))),
                    }
                }
                .shared();

                for ((p, _), permit) in planned[i..j].iter().zip(permits) {
                    let group_fetch = group_fetch.clone();
                    let rel = p.range.start - group_start..p.range.end - group_start;
                    let record_batch_idx = p.record_batch_idx;
                    let num_rows_this_rb = p.num_rows_this_rb;
                    let row_offset = p.row_offset;
                    let fetch_handle = ASYNC.spawn(async move {
                        let group_bytes = group_fetch
                            .await
                            .map_err(|e| polars_err!(ComputeError: "{e}"))?;
                        record_batch_data(
                            group_bytes.sliced(rel),
                            record_batch_idx,
                            num_rows_this_rb,
                            row_offset,
                        )
                        .await
                    });
                    let fetch_handle = tokio_handle_ext::AbortOnDropHandle(fetch_handle);

                    if prefetch_send
                        .send((fetch_handle, Some(permit)))
                        .await
                        .is_err()
                    {
                        break 'outer;
                    }
                }

                i = j;
                continue;
            }

            let (p, rb_fetch) = &planned[i];
            let range = p.range.clone();
            let record_batch_idx = p.record_batch_idx;
            let num_rows_this_rb = p.num_rows_this_rb;
            let row_offset = p.row_offset;

            let fetch_permit = match rb_fetch {
                RbFetch::All | RbFetch::Metadata => {
                    Some(pipeline_budget.acquire(range.len()).await)
                },
                RbFetch::None => None,
            };

            let current_byte_source = byte_source.clone();

            let fetch_handle = ASYNC.spawn(async move {
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
                    current_byte_source.get_range(range).await?
                };

                record_batch_data(
                    fetched_bytes,
                    record_batch_idx,
                    num_rows_this_rb,
                    row_offset,
                )
                .await
            });

            let fetch_handle = tokio_handle_ext::AbortOnDropHandle(fetch_handle);

            if prefetch_send
                .send((fetch_handle, fetch_permit))
                .await
                .is_err()
            {
                break;
            }

            i += 1;
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
