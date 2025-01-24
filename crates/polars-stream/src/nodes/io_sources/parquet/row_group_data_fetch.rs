use std::sync::Arc;

use polars_core::prelude::{ArrowSchema, PlHashMap};
use polars_core::series::IsSorted;
use polars_core::utils::operation_exceeded_idxsize_msg;
use polars_error::{polars_err, PolarsResult};
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::_internal::read_this_row_group;
use polars_io::prelude::{create_sorting_map, FileMetadata};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_parquet::read::RowGroupMetadata;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
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
    pub(super) sorting_map: PlHashMap<usize, IsSorted>,
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
    #[allow(unused)] // TODO: Fix!
    pub(super) predicate: Option<ScanIOPredicate>,
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
                let sorting_map = create_sorting_map(&row_group_metadata);

                self.current_row_offset = current_row_offset.saturating_add(num_rows);
                self.current_row_group_idx += 1;

                if self.use_statistics
                    && !match read_this_row_group(
                        self.predicate.as_ref(),
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
                                    memory_prefetch_func(unsafe { slice.get_unchecked(range) })
                                }
                            } else {
                                let range = row_group_metadata.full_byte_range();
                                let range = range.start as usize..range.end as usize;

                                memory_prefetch_func(unsafe { slice.get_unchecked(range) })
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
                        let mut ranges = get_row_group_byte_ranges_for_projection(
                            &row_group_metadata,
                            columns.as_ref(),
                        )
                        .collect::<Vec<_>>();

                        let n_ranges = ranges.len();

                        let bytes_map = current_byte_source.get_ranges(&mut ranges).await?;

                        assert_eq!(bytes_map.len(), n_ranges);

                        FetchedBytes::BytesMap(bytes_map)
                    } else {
                        // We still prefer `get_ranges()` over a single `get_range()` for downloading
                        // the entire row group, as it can have less memory-copying. A single `get_range()`
                        // would naively concatenate the memory blocks of the entire row group, while
                        // `get_ranges()` can skip concatenation since the downloaded blocks are
                        // aligned to the columns.
                        let mut ranges = row_group_metadata
                            .byte_ranges_iter()
                            .map(|x| x.start as usize..x.end as usize)
                            .collect::<Vec<_>>();

                        let n_ranges = ranges.len();

                        let bytes_map = current_byte_source.get_ranges(&mut ranges).await?;

                        assert_eq!(bytes_map.len(), n_ranges);

                        FetchedBytes::BytesMap(bytes_map)
                    };

                    PolarsResult::Ok(RowGroupData {
                        fetched_bytes,
                        path_index: current_path_index,
                        row_offset: current_row_offset,
                        slice,
                        file_max_row_group_height: current_max_row_group_height,
                        row_group_metadata,
                        sorting_map,
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

fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetadata,
    columns: &'a [PlSmallStr],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    columns.iter().flat_map(|col_name| {
        row_group_metadata
            .columns_under_root_iter(col_name)
            // `Option::into_iter` so that we return an empty iterator for the
            // `allow_missing_columns` case
            .into_iter()
            .flatten()
            .map(|col| {
                let byte_range = col.byte_range();
                byte_range.start as usize..byte_range.end as usize
            })
    })
}
