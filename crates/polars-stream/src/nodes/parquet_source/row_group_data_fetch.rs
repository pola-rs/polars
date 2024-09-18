use std::future::Future;
use std::sync::Arc;

use polars_core::prelude::{ArrowSchema, InitHashMaps, PlHashMap};
use polars_core::utils::operation_exceeded_idxsize_msg;
use polars_error::{polars_err, PolarsResult};
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::FileMetadata;
use polars_io::prelude::_internal::read_this_row_group;
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_parquet::read::RowGroupMetadata;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::IdxSize;

use super::mem_prefetch_funcs;
use super::row_group_decode::SharedFileState;
use crate::utils::task_handles_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RowGroupData {
    pub(super) fetched_bytes: FetchedBytes,
    pub(super) path_index: usize,
    pub(super) row_offset: usize,
    pub(super) slice: Option<(usize, usize)>,
    pub(super) file_max_row_group_height: usize,
    pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

pub(super) struct RowGroupDataFetcher {
    pub(super) metadata_rx: crate::async_primitives::connector::Receiver<(
        usize,
        usize,
        Arc<DynByteSource>,
        FileMetadata,
    )>,
    pub(super) use_statistics: bool,
    pub(super) verbose: bool,
    pub(super) reader_schema: Arc<ArrowSchema>,
    pub(super) projection: Option<Arc<[PlSmallStr]>>,
    pub(super) predicate: Option<Arc<dyn PhysicalIoExpr>>,
    pub(super) slice_range: Option<std::ops::Range<usize>>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) current_path_index: usize,
    pub(super) current_byte_source: Arc<DynByteSource>,
    pub(super) current_row_groups: std::vec::IntoIter<RowGroupMetadata>,
    pub(super) current_row_group_idx: usize,
    pub(super) current_max_row_group_height: usize,
    pub(super) current_row_offset: usize,
    pub(super) current_shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

impl RowGroupDataFetcher {
    pub(super) fn into_stream(self) -> RowGroupDataStream {
        RowGroupDataStream::new(self)
    }

    pub(super) async fn init_next_file_state(&mut self) -> bool {
        let Ok((path_index, row_offset, byte_source, metadata)) = self.metadata_rx.recv().await
        else {
            return false;
        };

        self.current_path_index = path_index;
        self.current_byte_source = byte_source;
        self.current_max_row_group_height = metadata.max_row_group_height;
        // The metadata task also sends a row offset to start counting from as it may skip files
        // during slice pushdown.
        self.current_row_offset = row_offset;
        self.current_row_group_idx = 0;
        self.current_row_groups = metadata.row_groups.into_iter();
        self.current_shared_file_state = Default::default();

        true
    }

    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        'main: loop {
            for row_group_metadata in self.current_row_groups.by_ref() {
                let current_row_offset = self.current_row_offset;
                let current_row_group_idx = self.current_row_group_idx;

                let num_rows = row_group_metadata.num_rows();

                self.current_row_offset = current_row_offset.saturating_add(num_rows);
                self.current_row_group_idx += 1;

                if self.use_statistics
                    && !match read_this_row_group(
                        self.predicate.as_deref(),
                        &row_group_metadata,
                        self.reader_schema.as_ref(),
                    ) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e)),
                    }
                {
                    if self.verbose {
                        eprintln!(
                            "[ParquetSource]: Predicate pushdown: \
                            Skipped row group {} in file {} ({} rows)",
                            current_row_group_idx, self.current_path_index, num_rows
                        );
                    }
                    continue;
                }

                if num_rows > IdxSize::MAX as usize {
                    let msg = operation_exceeded_idxsize_msg(
                        format!("number of rows in row group ({})", num_rows).as_str(),
                    );
                    return Some(Err(polars_err!(ComputeError: msg)));
                }

                let slice = if let Some(slice_range) = self.slice_range.clone() {
                    let (offset, len) = match SplitSlicePosition::split_slice_at_file(
                        current_row_offset,
                        num_rows,
                        slice_range,
                    ) {
                        SplitSlicePosition::Before => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Skipped row group {} in file {} ({} rows)",
                                    current_row_group_idx, self.current_path_index, num_rows
                                );
                            }
                            continue;
                        },
                        SplitSlicePosition::After => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Stop at row group {} in file {} \
                                    (remaining {} row groups will not be read)",
                                    current_row_group_idx,
                                    self.current_path_index,
                                    self.current_row_groups.len(),
                                );
                            };
                            break 'main;
                        },
                        SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    };

                    Some((offset, len))
                } else {
                    None
                };

                let current_byte_source = self.current_byte_source.clone();
                let projection = self.projection.clone();
                let current_shared_file_state = self.current_shared_file_state.clone();
                let memory_prefetch_func = self.memory_prefetch_func;
                let io_runtime = polars_io::pl_async::get_runtime();
                let current_path_index = self.current_path_index;
                let current_max_row_group_height = self.current_max_row_group_height;

                let handle = io_runtime.spawn(async move {
                    let fetched_bytes = if let DynByteSource::MemSlice(mem_slice) =
                        current_byte_source.as_ref()
                    {
                        // Skip byte range calculation for `no_prefetch`.
                        if memory_prefetch_func as usize != mem_prefetch_funcs::no_prefetch as usize
                        {
                            let slice = mem_slice.0.as_ref();

                            if let Some(columns) = projection.as_ref() {
                                for range in get_row_group_byte_ranges_for_projection(
                                    &row_group_metadata,
                                    columns.as_ref(),
                                ) {
                                    memory_prefetch_func(unsafe {
                                        slice.get_unchecked_release(range)
                                    })
                                }
                            } else {
                                let range = row_group_metadata.full_byte_range();
                                let range = range.start as usize..range.end as usize;

                                memory_prefetch_func(unsafe { slice.get_unchecked_release(range) })
                            };
                        }

                        // We have a mmapped or in-memory slice representing the entire
                        // file that can be sliced directly, so we can skip the byte-range
                        // calculations and HashMap allocation.
                        let mem_slice = mem_slice.0.clone();
                        FetchedBytes::MemSlice {
                            offset: 0,
                            mem_slice,
                        }
                    } else if let Some(columns) = projection.as_ref() {
                        let ranges = get_row_group_byte_ranges_for_projection(
                            &row_group_metadata,
                            columns.as_ref(),
                        )
                        .collect::<Vec<_>>();

                        let bytes = current_byte_source.get_ranges(ranges.as_ref()).await?;

                        assert_eq!(bytes.len(), ranges.len());

                        let mut bytes_map = PlHashMap::with_capacity(ranges.len());

                        for (range, bytes) in ranges.iter().zip(bytes) {
                            memory_prefetch_func(bytes.as_ref());
                            let v = bytes_map.insert(range.start, bytes);
                            debug_assert!(v.is_none(), "duplicate range start {}", range.start);
                        }

                        FetchedBytes::BytesMap(bytes_map)
                    } else {
                        // We have a dedicated code-path for a full projection that performs a
                        // single range request for the entire row group. During testing this
                        // provided much higher throughput from cloud than making multiple range
                        // request with `get_ranges()`.
                        let full_range = row_group_metadata.full_byte_range();
                        let full_range = full_range.start as usize..full_range.end as usize;

                        let mem_slice = {
                            let full_range_2 = full_range.clone();
                            task_handles_ext::AbortOnDropHandle(io_runtime.spawn(async move {
                                current_byte_source.get_range(full_range_2).await
                            }))
                            .await
                            .unwrap()?
                        };

                        FetchedBytes::MemSlice {
                            offset: full_range.start,
                            mem_slice,
                        }
                    };

                    PolarsResult::Ok(RowGroupData {
                        fetched_bytes,
                        path_index: current_path_index,
                        row_offset: current_row_offset,
                        slice,
                        file_max_row_group_height: current_max_row_group_height,
                        row_group_metadata,
                        shared_file_state: current_shared_file_state.clone(),
                    })
                });

                let handle = task_handles_ext::AbortOnDropHandle(handle);
                return Some(Ok(handle));
            }

            // Initialize state to the next file.
            if !self.init_next_file_state().await {
                break;
            }
        }

        None
    }
}

pub(super) enum FetchedBytes {
    MemSlice { mem_slice: MemSlice, offset: usize },
    BytesMap(PlHashMap<usize, MemSlice>),
}

impl FetchedBytes {
    pub(super) fn get_range(&self, range: std::ops::Range<usize>) -> MemSlice {
        match self {
            Self::MemSlice { mem_slice, offset } => {
                let offset = *offset;
                debug_assert!(range.start >= offset);
                mem_slice.slice(range.start - offset..range.end - offset)
            },
            Self::BytesMap(v) => {
                let v = v.get(&range.start).unwrap();
                debug_assert_eq!(v.len(), range.len());
                v.clone()
            },
        }
    }
}

#[rustfmt::skip]
type RowGroupDataStreamFut = std::pin::Pin<Box<
    dyn Future<
        Output =
            (
                Box<RowGroupDataFetcher>               ,
                Option                                 <
                PolarsResult                           <
                task_handles_ext::AbortOnDropHandle    <
                PolarsResult                           <
                RowGroupData      >      >      >      >
            )
    > + Send
>>;

pub(super) struct RowGroupDataStream {
    current_future: RowGroupDataStreamFut,
}

impl RowGroupDataStream {
    fn new(row_group_data_fetcher: RowGroupDataFetcher) -> Self {
        // [`RowGroupDataFetcher`] is a big struct, so we Box it once here to avoid boxing it on
        // every `next()` call.
        let current_future = Self::call_next_owned(Box::new(row_group_data_fetcher));
        Self { current_future }
    }

    fn call_next_owned(
        mut row_group_data_fetcher: Box<RowGroupDataFetcher>,
    ) -> RowGroupDataStreamFut {
        Box::pin(async move {
            let out = row_group_data_fetcher.next().await;
            (row_group_data_fetcher, out)
        })
    }
}

impl futures::stream::Stream for RowGroupDataStream {
    type Item = PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::pin::Pin;
        use std::task::Poll;

        match Pin::new(&mut self.current_future.as_mut()).poll(cx) {
            Poll::Ready((row_group_data_fetcher, out)) => {
                if out.is_some() {
                    self.current_future = Self::call_next_owned(row_group_data_fetcher);
                }

                Poll::Ready(out)
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetadata,
    columns: &'a [PlSmallStr],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    columns.iter().flat_map(|col_name| {
        row_group_metadata
            .columns_under_root_iter(col_name)
            .map(|col| {
                let byte_range = col.byte_range();
                byte_range.start as usize..byte_range.end as usize
            })
    })
}
