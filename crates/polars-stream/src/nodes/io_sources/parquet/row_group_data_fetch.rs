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

use crate::utils::task_handles_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RowGroupData {
    pub(super) fetched_bytes: FetchedBytes,
    pub(super) row_offset: usize,
    pub(super) slice: Option<(usize, usize)>,
    pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) sorting_map: PlHashMap<usize, IsSorted>,
}

pub(super) struct RowGroupDataFetcher {
    pub(super) use_statistics: bool,
    pub(super) verbose: bool,
    pub(super) reader_schema: Arc<ArrowSchema>,
    pub(super) projection: Option<Arc<[PlSmallStr]>>,
    #[allow(unused)] // TODO: Fix!
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) slice_range: Option<std::ops::Range<usize>>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,
    pub(super) row_group_idx: usize,
    pub(super) row_offset: usize,
}

impl RowGroupDataFetcher {
    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        while self.row_group_idx < self.metadata.row_groups.len() {
            let idx = self.row_group_idx;
            let row_group_metadata = &self.metadata.row_groups[idx];
            let current_row_offset = self.row_offset;
            let current_row_group_idx = self.row_group_idx;

            let num_rows = row_group_metadata.num_rows();
            let sorting_map = create_sorting_map(&row_group_metadata);

            self.row_offset = current_row_offset.saturating_add(num_rows);
            self.row_group_idx += 1;

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
                            Skipped row group {} in file ({} rows)",
                        current_row_group_idx, num_rows
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
                                    Skipped row group {} ({} rows)",
                                current_row_group_idx, num_rows
                            );
                        }
                        continue;
                    },
                    SplitSlicePosition::After => {
                        if self.verbose {
                            eprintln!(
                                "[ParquetSource]: Slice pushdown: \
                                    Stop at row group {} \
                                    (remaining {} row groups will not be read)",
                                current_row_group_idx,
                                self.metadata.row_groups.len() - self.row_group_idx,
                            );
                        };
                        return None;
                    },
                    SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                };

                Some((offset, len))
            } else {
                None
            };

            let metadata = self.metadata.clone();
            let current_byte_source = self.byte_source.clone();
            let projection = self.projection.clone();
            let memory_prefetch_func = self.memory_prefetch_func;
            let io_runtime = polars_io::pl_async::get_runtime();

            let handle = io_runtime.spawn(async move {
                let row_group_metadata = &metadata.row_groups[idx];
                let fetched_bytes =
                    if let DynByteSource::MemSlice(mem_slice) = current_byte_source.as_ref() {
                        // Skip byte range calculation for `no_prefetch`.
                        if memory_prefetch_func as usize
                            != polars_utils::mem::prefetch::no_prefetch as usize
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
                    row_offset: current_row_offset,
                    slice,
                    // @TODO: Remove clone
                    row_group_metadata: row_group_metadata.clone(),
                    sorting_map,
                })
            });

            let handle = task_handles_ext::AbortOnDropHandle(handle);
            return Some(Ok(handle));
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
