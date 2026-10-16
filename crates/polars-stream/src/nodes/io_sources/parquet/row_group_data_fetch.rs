use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_buffer::Buffer;
use polars_core::prelude::PlHashMap;
use polars_core::runtime::ASYNC;
use polars_core::series::IsSorted;
use polars_core::utils::polars_arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_ensure};
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, create_sorting_map};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::read::RowGroupMetadata;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;
use crate::nodes::io_sources::parquet::statistics::RuntimeRangeSkipper;
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
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) metadata: Arc<FileMetadata>,
    pub(super) byte_source: Arc<DynByteSource>,

    /// The row groups to read, in the order to read them.
    pub(super) row_groups: VecDeque<RowGroupToRead>,
    pub(super) runtime_range_skipper: Option<Arc<RuntimeRangeSkipper>>,

    pub(super) read_stats: Arc<ReadStats>,
}

pub(super) struct RowGroupToRead {
    pub(super) idx: usize,
    /// The row offset of its first row in the file.
    pub(super) row_offset: usize,
    pub(super) slice: Option<(usize, usize)>,
}

/// The row groups of `row_group_slice` that `row_group_mask` keeps, in file order. The
/// first starts at `row_offset` and `slice_range` is relative to it.
pub(super) fn row_groups_to_read(
    metadata: &FileMetadata,
    row_group_slice: Range<usize>,
    row_group_mask: Option<&Bitmap>,
    mut row_offset: usize,
    mut slice_range: Option<Range<usize>>,
) -> VecDeque<RowGroupToRead> {
    let mut row_groups = VecDeque::with_capacity(row_group_slice.len());
    for (i, idx) in row_group_slice.enumerate() {
        let num_rows = metadata.row_groups[idx].num_rows();
        let slice = slice_range.as_mut().map(|slice_range| {
            let rg_row_start = slice_range.start;
            let rg_row_end = slice_range.end.min(num_rows);

            *slice_range = slice_range.start.saturating_sub(num_rows)
                ..slice_range.end.saturating_sub(num_rows);

            (rg_row_start, rg_row_end - rg_row_start)
        });
        if !row_group_mask.is_some_and(|mask| mask.get_bit(i)) {
            row_groups.push_back(RowGroupToRead {
                idx,
                row_offset,
                slice,
            });
        }
        row_offset = row_offset.saturating_add(num_rows);
    }
    row_groups
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
    /// Moves past the row groups at the front that the runtime ranges skip.
    fn skip_front(&mut self) {
        while let Some(row_group) = self.row_groups.front()
            && self
                .runtime_range_skipper
                .as_ref()
                .is_some_and(|s| s.skips(&self.metadata.row_groups[row_group.idx]))
        {
            self.row_groups.pop_front();
        }
    }

    /// Returns the projected byte size of the next row group to be fetched, without spawning
    /// any I/O. Returns None if there are no more row groups.
    pub(super) fn peek_next_bytes(&mut self) -> Option<u64> {
        self.skip_front();
        let row_group_metadata = &self.metadata.row_groups[self.row_groups.front()?.idx];

        let n_bytes = match self.byte_source.as_ref() {
            DynByteSource::Buffer(_) => 0, // in-memory, no budget needed
            // Ignore any errors getting the byte range. If there are columns without
            // metadata available (due to missing a decryption key) then an error will
            // be raised later when reading them.
            _ if !self.is_full_projection => get_row_group_byte_ranges_for_projection(
                row_group_metadata,
                &mut self.projection.iter().map(|x| &x.arrow_field().name),
            )
            .filter_map(|r| r.ok())
            .map(|r| r.len() as u64)
            .sum(),
            _ => row_group_metadata
                .byte_ranges_iter()
                .map(|x| x.end - x.start)
                .sum(),
        };

        Some(n_bytes)
    }

    /// Fetches the row group `peek_next_bytes` sized.
    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<tokio_handle_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        let RowGroupToRead {
            idx,
            row_offset: current_row_offset,
            slice,
        } = self.row_groups.pop_front()?;
        let sorting_map = create_sorting_map(&self.metadata.row_groups[idx]);

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
                            // Errors are raised when decoding rather than when prefetching.
                            for range in get_row_group_byte_ranges_for_projection(
                                row_group_metadata,
                                &mut projection.iter().map(|x| &x.arrow_field().name),
                            )
                            .filter_map(|r| r.ok())
                            {
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
                    let mut ranges = get_row_group_fetch_ranges(
                        row_group_metadata,
                        &projection,
                        is_full_projection,
                    )?;

                    if ranges.is_empty() {
                        // Nothing to read, e.g. an empty row group.
                        FetchedBytes::BytesMap(PlHashMap::default())
                    } else if read_stats.defer_cached_reads && source.is_cached(&ranges) {
                        // Read a cached row group in its decode tasks, each column chunk on
                        // the thread that decompresses it. This beats copying it ahead on the blocking
                        // pool. Uncached ones are prefetched, so that the I/O overlaps decoding.
                        read_stats.deferred.fetch_add(1, Ordering::Relaxed);
                        FetchedBytes::Deferred {
                            source: current_byte_source.clone(),
                            ranges,
                            read_stats: read_stats.clone(),
                        }
                    } else {
                        read_stats.prefetched.fetch_add(1, Ordering::Relaxed);

                        FetchedBytes::BytesMap(
                            fetch_ranges(&current_byte_source, &mut ranges).await?,
                        )
                    }
                },
                DynByteSource::Cloud(_) => {
                    // We still prefer `get_ranges()` over a single `get_range()` for downloading
                    // the entire row group, as it can have less memory-copying. A single `get_range()`
                    // would naively concatenate the memory blocks of the entire row group, while
                    // `get_ranges()` can skip concatenation since the downloaded blocks are
                    // aligned to the columns.
                    let mut ranges = get_row_group_fetch_ranges(
                        row_group_metadata,
                        &projection,
                        is_full_projection,
                    )?;

                    FetchedBytes::BytesMap(fetch_ranges(&current_byte_source, &mut ranges).await?)
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
        Some(Ok(handle))
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
        let bytes_map = tokio_handle_ext::AbortOnDropHandle(
            ASYNC.spawn(async move { fetch_ranges(&source, &mut ranges).await }),
        )
        .await
        .expect("fetch task panicked")?;

        *self = Self::BytesMap(bytes_map);
        Ok(())
    }

    pub(super) fn get_range(&self, range: std::ops::Range<usize>) -> PolarsResult<Buffer<u8>> {
        // Empty ranges are not fetched.
        if range.is_empty() {
            return Ok(Buffer::new());
        }

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

/// Byte ranges of the projected column chunks to fetch. Empty ranges (chunks without pages) are
/// left out: they need no I/O and can share their start with another range.
fn get_row_group_fetch_ranges(
    row_group_metadata: &RowGroupMetadata,
    projection: &[ArrowFieldProjection],
    is_full_projection: bool,
) -> ParquetResult<Vec<Range<usize>>> {
    let mut ranges = if !is_full_projection {
        get_row_group_byte_ranges_for_projection(
            row_group_metadata,
            &mut projection.iter().map(|x| &x.arrow_field().name),
        )
        .collect::<ParquetResult<Vec<_>>>()?
    } else {
        row_group_metadata
            .byte_ranges_iter()
            .map(|x| x.start as usize..x.end as usize)
            .collect::<Vec<_>>()
    };
    ranges.retain(|r| !r.is_empty());
    Ok(ranges)
}

/// Fetches `ranges`, keyed by their start.
async fn fetch_ranges(
    source: &DynByteSource,
    ranges: &mut [Range<usize>],
) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
    let n_ranges = ranges.len();
    let bytes_map = source.get_ranges(ranges).await?;
    polars_ensure!(
        bytes_map.len() == n_ranges,
        ComputeError: "parquet column chunks share a start offset: {} distinct of {} byte ranges",
        bytes_map.len(), n_ranges
    );
    Ok(bytes_map)
}

fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetadata,
    columns: &'a mut dyn Iterator<Item = &PlSmallStr>,
) -> impl Iterator<Item = ParquetResult<std::ops::Range<usize>>> + 'a {
    columns.flat_map(|col_name| {
        row_group_metadata
            .columns_under_root_iter(col_name)
            // `Option::into_iter` so that we return an empty iterator for the
            // `allow_missing_columns` case
            .into_iter()
            .flatten()
            .map(|col| {
                let byte_range = col.byte_range()?;
                Ok(byte_range.start as usize..byte_range.end as usize)
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
