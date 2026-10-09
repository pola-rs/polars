use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_buffer::Buffer;
use polars_core::prelude::PlHashMap;
use polars_core::runtime::ASYNC;
use polars_core::series::IsSorted;
use polars_core::utils::polars_arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, create_sorting_map};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::read::RowGroupMetadata;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;
use crate::utils::tokio_handle_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RowGroupData {
    pub(super) fetched_bytes: FetchedBytes,
    pub(super) row_offset: usize,
    pub(super) slice: Option<(usize, usize)>,
    pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) sorting_map: Vec<(usize, IsSorted)>,
}

pub(super) struct RowGroupDataFetcher {
    pub(super) projection: Arc<[ArrowFieldProjection]>,
    pub(super) is_full_projection: bool,
    #[allow(unused)] // TODO: Fix!
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) slice_range: Option<Range<usize>>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,

    pub(super) row_group_slice: Range<usize>,
    pub(super) row_group_mask: Option<Bitmap>,

    pub(super) row_offset: usize,

    pub(super) read_stats: Arc<ReadStats>,
}

/// How the row groups of a local file were read, and `POLARS_FILE_DEFER_CACHED_READS`. Logged
/// when the last row group holding it has been decoded.
pub(super) struct ReadStats {
    verbose: bool,
    defer_cached_reads: bool,
    deferred: AtomicUsize,
    prefetched: AtomicUsize,
    /// Deferred row groups whose pages were evicted before their decode.
    reread: AtomicUsize,
}

impl ReadStats {
    pub(super) fn new(verbose: bool, defer_cached_reads: bool) -> Self {
        Self {
            verbose,
            defer_cached_reads,
            deferred: AtomicUsize::new(0),
            prefetched: AtomicUsize::new(0),
            reread: AtomicUsize::new(0),
        }
    }
}

impl Drop for ReadStats {
    fn drop(&mut self) {
        let deferred = *self.deferred.get_mut();
        let total = deferred + *self.prefetched.get_mut();
        let reread = *self.reread.get_mut();
        if self.verbose && total > 0 {
            let reread = if reread > 0 {
                format!(", {reread} re-read after eviction")
            } else {
                String::new()
            };
            let disabled = if self.defer_cached_reads {
                ""
            } else {
                " (disabled)"
            };
            eprintln!(
                "[ParquetFileReader]: Deferred cached reads: {deferred} / {total} row groups\
                {reread}{disabled}"
            );
        }
    }
}

impl RowGroupDataFetcher {
    /// Returns the projected byte size of the next row group to be fetched, without advancing
    /// state or spawning any I/O. Returns None if there are no more row groups.
    pub(super) fn peek_next_bytes(&self) -> Option<u64> {
        // Walk forward from current position to find the next unmasked row group
        let mut slice_start = self.row_group_slice.start;
        let mut mask_offset = 0;

        while slice_start < self.row_group_slice.end {
            // Check mask
            if let Some(mask) = &self.row_group_mask {
                if mask.get_bit(mask_offset) {
                    // masked out, skip
                    slice_start += 1;
                    mask_offset += 1;
                    continue;
                }
            }

            let row_group_metadata = &self.metadata.row_groups[slice_start];

            let n_bytes = match self.byte_source.as_ref() {
                DynByteSource::Buffer(_) => 0, // in-memory, no budget needed
                _ if !self.is_full_projection => get_row_group_byte_ranges_for_projection(
                    row_group_metadata,
                    &mut self.projection.iter().map(|x| &x.arrow_field().name),
                )
                .map(|r| r.len() as u64)
                .sum(),
                _ => row_group_metadata
                    .byte_ranges_iter()
                    .map(|x| x.end - x.start)
                    .sum(),
            };

            return Some(n_bytes);
        }

        None
    }

    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<tokio_handle_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        while !self.row_group_slice.is_empty() {
            let idx = self.row_group_slice.start;
            self.row_group_slice.start += 1;

            let row_group_metadata = &self.metadata.row_groups[idx];
            let current_row_offset = self.row_offset;

            let num_rows = row_group_metadata.num_rows();
            let sorting_map = create_sorting_map(row_group_metadata);

            self.row_offset = current_row_offset.saturating_add(num_rows);

            let slice = if let Some(slice_range) = self.slice_range.as_mut() {
                let rg_row_start = slice_range.start;
                let rg_row_end = slice_range.end.min(num_rows);

                *slice_range = slice_range.start.saturating_sub(num_rows)
                    ..slice_range.end.saturating_sub(num_rows);

                Some((rg_row_start, rg_row_end - rg_row_start))
            } else {
                None
            };

            if let Some(row_group_mask) = self.row_group_mask.as_mut() {
                let do_skip = row_group_mask.get_bit(0);
                row_group_mask.slice(1, self.row_group_slice.len());

                if do_skip {
                    continue;
                }
            }

            let metadata = self.metadata.clone();
            let current_byte_source = self.byte_source.clone();
            let projection = self.projection.clone();
            let is_full_projection = self.is_full_projection;
            let memory_prefetch_func = self.memory_prefetch_func;
            let read_stats = self.read_stats.clone();

            let handle = ASYNC.spawn(async move {
                let row_group_metadata = &metadata.row_groups[idx];
                let fetched_bytes = match current_byte_source.as_ref() {
                    DynByteSource::Buffer(mem_slice) => {
                        // Skip byte range calculation for `no_prefetch`.
                        if memory_prefetch_func as usize
                            != polars_utils::mem::prefetch::no_prefetch as *const () as usize
                        {
                            let slice = mem_slice.0.as_ref();

                            if !is_full_projection {
                                for range in get_row_group_byte_ranges_for_projection(
                                    row_group_metadata,
                                    &mut projection.iter().map(|x| &x.arrow_field().name),
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
                        FetchedBytes::Buffer {
                            offset: 0,
                            buffer: mem_slice,
                        }
                    },
                    DynByteSource::File(source) => {
                        let mut ranges = if !is_full_projection {
                            get_row_group_byte_ranges_for_projection(
                                row_group_metadata,
                                &mut projection.iter().map(|x| &x.arrow_field().name),
                            )
                            .collect::<Vec<_>>()
                        } else {
                            row_group_metadata
                                .byte_ranges_iter()
                                .map(|x| x.start as usize..x.end as usize)
                                .collect::<Vec<_>>()
                        };

                        // Reade a cached row group in its decode tasks, each column chunk on
                        // the thread that decompresses it. This beats copying it ahead on the blocking
                        // pool. Uncached ones are prefetched, so that the I/O overlaps decoding.
                        let defer = read_stats.defer_cached_reads
                            && !ranges.is_empty()
                            && source.is_cached(&ranges);

                        if defer {
                            read_stats.deferred.fetch_add(1, Ordering::Relaxed);
                            FetchedBytes::Deferred {
                                source: current_byte_source.clone(),
                                ranges,
                                read_stats: read_stats.clone(),
                            }
                        } else {
                            read_stats.prefetched.fetch_add(1, Ordering::Relaxed);

                            let n_ranges = ranges.len();

                            let bytes_map = source.get_ranges(&mut ranges).await?;

                            assert_eq!(bytes_map.len(), n_ranges);

                            FetchedBytes::BytesMap(bytes_map)
                        }
                    },
                    DynByteSource::Cloud(_) => {
                        if !is_full_projection {
                            let mut ranges = get_row_group_byte_ranges_for_projection(
                                row_group_metadata,
                                &mut projection.iter().map(|x| &x.arrow_field().name),
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
                        }
                    },
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

            let handle = tokio_handle_ext::AbortOnDropHandle(handle);
            return Some(Ok(handle));
        }

        None
    }
}

pub(super) enum FetchedBytes {
    Buffer {
        buffer: Buffer<u8>,
        offset: usize,
    },
    BytesMap(PlHashMap<usize, Buffer<u8>>),
    /// Nothing fetched: the row group was cached, so `get_range` reads from this file source on
    /// the calling (decode) thread.
    Deferred {
        source: Arc<DynByteSource>,
        ranges: Vec<Range<usize>>,
        read_stats: Arc<ReadStats>,
    },
}

impl FetchedBytes {
    /// Re-checks a deferred row group right before its decode, because its pages can be evicted
    /// since the last cache check.
    pub(super) async fn fetch_if_evicted(&mut self) -> PolarsResult<()> {
        let Self::Deferred {
            source,
            ranges,
            read_stats,
        } = self
        else {
            return Ok(());
        };
        let DynByteSource::File(file) = source.as_ref() else {
            unreachable!("only file sources defer their reads")
        };
        if file.is_cached(ranges) {
            return Ok(());
        }

        read_stats.reread.fetch_add(1, Ordering::Relaxed);
        let source = source.clone();
        let mut ranges = std::mem::take(ranges);
        let n_ranges = ranges.len();
        let bytes_map = tokio_handle_ext::AbortOnDropHandle(
            ASYNC.spawn(async move { source.get_ranges(&mut ranges).await }),
        )
        .await
        .expect("fetch task panicked")?;
        assert_eq!(bytes_map.len(), n_ranges);

        *self = Self::BytesMap(bytes_map);
        Ok(())
    }

    pub(super) fn get_range(&self, range: std::ops::Range<usize>) -> PolarsResult<Buffer<u8>> {
        Ok(match self {
            Self::Buffer { buffer, offset } => {
                let offset = *offset;
                debug_assert!(range.start >= offset);
                buffer
                    .clone()
                    .sliced(range.start - offset..range.end - offset)
            },
            Self::BytesMap(v) => {
                let v = v.get(&range.start).unwrap();
                debug_assert_eq!(v.len(), range.len());
                v.clone()
            },
            Self::Deferred { source, .. } => match source.as_ref() {
                DynByteSource::File(source) => source.read_blocking(range)?,
                _ => unreachable!("only file sources defer their reads"),
            },
        })
    }
}

fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetadata,
    columns: &'a mut dyn Iterator<Item = &PlSmallStr>,
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    columns.flat_map(|col_name| {
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

#[cfg(all(
    test,
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use std::os::fd::AsRawFd;

    use polars_config::FileAdvice;
    use polars_io::utils::byte_source::{FileByteSource, FileReadContext};
    use tokio::sync::Semaphore;

    use super::*;

    #[test]
    fn deferred_row_group_is_read_again_after_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        let contents: Vec<u8> = (0..1 << 20).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &contents).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let read_context = FileReadContext {
            enable_o_direct: false,
            concurrency: 4,
            permits: Arc::new(Semaphore::new(4)),
            advice: FileAdvice::Normal,
        };
        let source = Arc::new(DynByteSource::from(
            FileByteSource::try_new_from_std(file.try_clone().unwrap(), read_context, None)
                .unwrap(),
        ));
        let DynByteSource::File(file_source) = source.as_ref() else {
            unreachable!()
        };
        let ranges = vec![0..1000, 4096..70_000];
        let read_stats = Arc::new(ReadStats::new(false, true));
        let deferred = || FetchedBytes::Deferred {
            source: source.clone(),
            ranges: ranges.clone(),
            read_stats: read_stats.clone(),
        };

        // Just written, so cached; `false` means the kernel lacks cachestat(2).
        if !file_source.is_cached(&ranges) {
            return;
        }
        let mut fetched = deferred();
        ASYNC.block_on(fetched.fetch_if_evicted()).unwrap();
        assert!(matches!(fetched, FetchedBytes::Deferred { .. }));

        // Evicted after the fetch saw it cached. tmpfs keeps its pages, so only check where
        // they can go.
        file.sync_all().unwrap();
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if file_source.is_cached(&ranges) {
            return;
        }
        let mut fetched = deferred();
        ASYNC.block_on(fetched.fetch_if_evicted()).unwrap();
        assert!(matches!(fetched, FetchedBytes::BytesMap(_)));
        for r in &ranges {
            assert_eq!(
                fetched.get_range(r.clone()).unwrap().as_ref(),
                &contents[r.clone()]
            );
        }
        assert_eq!(read_stats.reread.load(Ordering::Relaxed), 1);
    }
}
