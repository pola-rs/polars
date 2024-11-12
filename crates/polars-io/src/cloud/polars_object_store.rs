use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_error::{to_compute_err, PolarsError, PolarsResult};
use tokio::io::AsyncWriteExt;

use crate::pl_async::{
    self, get_concurrency_limit, get_download_chunk_size, tune_with_concurrency_budget,
    with_concurrency_budget, MAX_BUDGET_PER_REQUEST,
};

/// Polars specific wrapper for `Arc<dyn ObjectStore>` that limits the number of
/// concurrent requests for the entire application.
#[derive(Debug, Clone)]
pub struct PolarsObjectStore(Arc<dyn ObjectStore>);
pub type ObjectStorePath = object_store::path::Path;

impl PolarsObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self(store)
    }

    /// Returns a buffered stream that downloads concurrently up to the concurrency limit.
    fn get_buffered_ranges_stream<'a, T: Iterator<Item = Range<usize>>>(
        &'a self,
        path: &'a Path,
        ranges: T,
    ) -> impl StreamExt<Item = PolarsResult<Bytes>>
           + TryStreamExt<Ok = Bytes, Error = PolarsError, Item = PolarsResult<Bytes>>
           + use<'a, T> {
        futures::stream::iter(
            ranges
                .map(|range| async { self.0.get_range(path, range).await.map_err(to_compute_err) }),
        )
        // Add a limit locally as this gets run inside a single `tune_with_concurrency_budget`.
        .buffered(get_concurrency_limit() as usize)
    }

    pub async fn get_range(&self, path: &Path, range: Range<usize>) -> PolarsResult<Bytes> {
        let parts = split_range(range.clone());

        if parts.len() == 1 {
            tune_with_concurrency_budget(1, || self.0.get_range(path, range))
                .await
                .map_err(to_compute_err)
        } else {
            let parts = tune_with_concurrency_budget(
                parts.len().clamp(0, MAX_BUDGET_PER_REQUEST) as u32,
                || {
                    self.get_buffered_ranges_stream(path, parts)
                        .try_collect::<Vec<Bytes>>()
                },
            )
            .await?;

            let mut combined = Vec::with_capacity(range.len());

            for part in parts {
                combined.extend_from_slice(&part)
            }

            assert_eq!(combined.len(), range.len());

            PolarsResult::Ok(Bytes::from(combined))
        }
    }

    /// Fetch byte ranges into a HashMap keyed by the range start. This will mutably sort the
    /// `ranges` slice for coalescing.
    ///
    /// # Panics
    /// Panics if the same range start is used by more than 1 range.
    pub async fn get_ranges_sort<
        K: TryFrom<usize, Error = impl std::fmt::Debug> + std::hash::Hash + Eq,
        T: From<Bytes>,
    >(
        &self,
        path: &Path,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<K, T>> {
        if ranges.is_empty() {
            return Ok(Default::default());
        }

        let mut out = PlHashMap::with_capacity(ranges.len());

        ranges.sort_unstable_by_key(|x| x.start);

        let (merged_ranges, merged_ends): (Vec<_>, Vec<_>) = merge_ranges(ranges).unzip();

        let mut stream = self.get_buffered_ranges_stream(path, merged_ranges.iter().cloned());

        tune_with_concurrency_budget(
            merged_ranges.len().clamp(0, MAX_BUDGET_PER_REQUEST) as u32,
            || async {
                let mut len = 0;
                let mut current_offset = 0;
                let mut ends_iter = merged_ends.iter();

                let mut splitted_parts = vec![];

                while let Some(bytes) = stream.try_next().await? {
                    len += bytes.len();
                    let end = *ends_iter.next().unwrap();

                    if end == 0 {
                        splitted_parts.push(bytes);
                        continue;
                    }

                    let full_range = ranges[current_offset..end]
                        .iter()
                        .cloned()
                        .reduce(|l, r| l.start.min(r.start)..l.end.max(r.end))
                        .unwrap();

                    let bytes = if splitted_parts.is_empty() {
                        bytes
                    } else {
                        let mut out = Vec::with_capacity(full_range.len());

                        for x in splitted_parts.drain(..) {
                            out.extend_from_slice(&x);
                        }

                        out.extend_from_slice(&bytes);
                        Bytes::from(out)
                    };

                    assert_eq!(bytes.len(), full_range.len());

                    for range in &ranges[current_offset..end] {
                        let v = out.insert(
                            K::try_from(range.start).unwrap(),
                            T::from(bytes.slice(
                                range.start - full_range.start..range.end - full_range.start,
                            )),
                        );

                        assert!(v.is_none()); // duplicate range start
                    }

                    current_offset = end;
                }

                assert!(splitted_parts.is_empty());

                PolarsResult::Ok(pl_async::Size::from(len as u64))
            },
        )
        .await?;

        Ok(out)
    }

    pub async fn download(&self, path: &Path, file: &mut tokio::fs::File) -> PolarsResult<()> {
        let opt_size = self.head(path).await.ok().map(|x| x.size);
        let parts = opt_size.map(|x| split_range(0..x)).filter(|x| x.len() > 1);

        if let Some(parts) = parts {
            tune_with_concurrency_budget(
                parts.len().clamp(0, MAX_BUDGET_PER_REQUEST) as u32,
                || async {
                    let mut stream = self.get_buffered_ranges_stream(path, parts);
                    let mut len = 0;
                    while let Some(bytes) = stream.try_next().await? {
                        len += bytes.len();
                        file.write_all(&bytes).await.map_err(to_compute_err)?;
                    }

                    assert_eq!(len, opt_size.unwrap());

                    PolarsResult::Ok(pl_async::Size::from(len as u64))
                },
            )
            .await?
        } else {
            tune_with_concurrency_budget(1, || async {
                let mut stream = self
                    .0
                    .get(path)
                    .await
                    .map_err(to_compute_err)?
                    .into_stream();

                let mut len = 0;
                while let Some(bytes) = stream.try_next().await? {
                    len += bytes.len();
                    file.write_all(&bytes).await.map_err(to_compute_err)?;
                }

                PolarsResult::Ok(pl_async::Size::from(len as u64))
            })
            .await?
        };

        // Dropping is delayed for tokio async files so we need to explicitly
        // flush here (https://github.com/tokio-rs/tokio/issues/2307#issuecomment-596336451).
        file.sync_all().await.map_err(PolarsError::from)?;

        Ok(())
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(&self, path: &Path) -> PolarsResult<ObjectMeta> {
        with_concurrency_budget(1, || async {
            let head_result = self.0.head(path).await;

            if head_result.is_err() {
                // Pre-signed URLs forbid the HEAD method, but we can still retrieve the header
                // information with a range 0-0 request.
                let get_range_0_0_result = self
                    .0
                    .get_opts(
                        path,
                        object_store::GetOptions {
                            range: Some((0..1).into()),
                            ..Default::default()
                        },
                    )
                    .await;

                if let Ok(v) = get_range_0_0_result {
                    return Ok(v.meta);
                }
            }

            head_result
        })
        .await
        .map_err(to_compute_err)
    }
}

/// Splits a single range into multiple smaller ranges, which can be downloaded concurrently for
/// much higher throughput.
fn split_range(range: Range<usize>) -> impl ExactSizeIterator<Item = Range<usize>> {
    let chunk_size = get_download_chunk_size();

    // Calculate n_parts such that we are as close as possible to the `chunk_size`.
    let n_parts = [
        (range.len().div_ceil(chunk_size)).max(1),
        (range.len() / chunk_size).max(1),
    ]
    .into_iter()
    .min_by_key(|x| (range.len() / *x).abs_diff(chunk_size))
    .unwrap();

    let chunk_size = (range.len() / n_parts).max(1);

    assert_eq!(n_parts, (range.len() / chunk_size).max(1));
    let bytes_rem = range.len() % chunk_size;

    (0..n_parts).map(move |part_no| {
        let (start, end) = if part_no == 0 {
            // Download remainder length in the first chunk since it starts downloading first.
            let end = range.start + chunk_size + bytes_rem;
            let end = if end > range.end { range.end } else { end };
            (range.start, end)
        } else {
            let start = bytes_rem + range.start + part_no * chunk_size;
            (start, start + chunk_size)
        };

        start..end
    })
}

/// Note: For optimal performance, `ranges` should be sorted. More generally,
/// ranges placed next to each other should also be close in range value.
///
/// # Returns
/// `[(range1, end1), (range2, end2)]`, where:
/// * `range1` contains bytes for the ranges from `ranges[0..end1]`
/// * `range2` contains bytes for the ranges from `ranges[end1..end2]`
/// * etc..
///
/// Note that if an end value is 0, it means the range is a splitted part and should be combined.
fn merge_ranges(ranges: &[Range<usize>]) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
    let chunk_size = get_download_chunk_size();

    let mut current_merged_range = ranges.first().map_or(0..0, Clone::clone);
    // Number of fetched bytes excluding excess.
    let mut current_n_bytes = current_merged_range.len();

    (0..ranges.len())
        .filter_map(move |current_idx| {
            let current_idx = 1 + current_idx;

            if current_idx == ranges.len() {
                // No more items - flush current state.
                Some((current_merged_range.clone(), current_idx))
            } else {
                let range = ranges[current_idx].clone();

                let new_merged = current_merged_range.start.min(range.start)
                    ..current_merged_range.end.max(range.end);

                // E.g.:
                // |--------|
                //  oo        // range1
                //       oo   // range2
                //    ^^^     // distance = 3, is_overlapping = false
                // E.g.:
                // |--------|
                //  ooooo     // range1
                //     ooooo  // range2
                //     ^^     // distance = 2, is_overlapping = true
                let (distance, is_overlapping) = {
                    let l = current_merged_range.end.min(range.end);
                    let r = current_merged_range.start.max(range.start);

                    (r.abs_diff(l), r < l)
                };

                let should_merge = is_overlapping || {
                    let leq_current_len_dist_to_chunk_size = new_merged.len().abs_diff(chunk_size)
                        <= current_merged_range.len().abs_diff(chunk_size);
                    let gap_tolerance =
                        (current_n_bytes.max(range.len()) / 8).clamp(1024 * 1024, 8 * 1024 * 1024);

                    leq_current_len_dist_to_chunk_size && distance <= gap_tolerance
                };

                if should_merge {
                    // Merge to existing range
                    current_merged_range = new_merged;
                    current_n_bytes += if is_overlapping {
                        range.len() - distance
                    } else {
                        range.len()
                    };
                    None
                } else {
                    let out = (current_merged_range.clone(), current_idx);
                    current_merged_range = range;
                    current_n_bytes = current_merged_range.len();
                    Some(out)
                }
            }
        })
        .flat_map(|x| {
            // Split large individual ranges within the list of ranges.
            let (range, end) = x;
            let split = split_range(range.clone());
            let len = split.len();

            split
                .enumerate()
                .map(move |(i, range)| (range, if 1 + i == len { end } else { 0 }))
        })
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_split_range() {
        use super::{get_download_chunk_size, split_range};

        let chunk_size = get_download_chunk_size();

        assert_eq!(chunk_size, 64 * 1024 * 1024);

        #[allow(clippy::single_range_in_vec_init)]
        {
            // Round-trip empty ranges.
            assert_eq!(split_range(0..0).collect::<Vec<_>>(), [0..0]);
            assert_eq!(split_range(3..3).collect::<Vec<_>>(), [3..3]);
        }

        // Threshold to start splitting to 2 ranges
        //
        // n - chunk_size == chunk_size - n / 2
        // n + n / 2 == 2 * chunk_size
        // 3 * n == 4 * chunk_size
        // n = 4 * chunk_size / 3
        let n = 4 * chunk_size / 3;

        #[allow(clippy::single_range_in_vec_init)]
        {
            assert_eq!(split_range(0..n).collect::<Vec<_>>(), [0..89478485]);
        }

        assert_eq!(
            split_range(0..n + 1).collect::<Vec<_>>(),
            [0..44739243, 44739243..89478486]
        );

        // Threshold to start splitting to 3 ranges
        //
        // n / 2 - chunk_size == chunk_size - n / 3
        // n / 2 + n / 3 == 2 * chunk_size
        // 5 * n == 12 * chunk_size
        // n == 12 * chunk_size / 5
        let n = 12 * chunk_size / 5;

        assert_eq!(
            split_range(0..n).collect::<Vec<_>>(),
            [0..80530637, 80530637..161061273]
        );

        assert_eq!(
            split_range(0..n + 1).collect::<Vec<_>>(),
            [0..53687092, 53687092..107374183, 107374183..161061274]
        );
    }

    #[test]
    fn test_merge_ranges() {
        use super::{get_download_chunk_size, merge_ranges};

        let chunk_size = get_download_chunk_size();

        assert_eq!(chunk_size, 64 * 1024 * 1024);

        // Round-trip empty slice
        assert_eq!(merge_ranges(&[]).collect::<Vec<_>>(), []);

        // We have 1 tiny request followed by 1 huge request. They are combined as it reduces the
        // `abs_diff()` to the `chunk_size`, but afterwards they are split to 2 evenly sized
        // requests.
        assert_eq!(
            merge_ranges(&[0..1, 1..127 * 1024 * 1024]).collect::<Vec<_>>(),
            [(0..66584576, 0), (66584576..133169152, 2)]
        );

        // <= 1MiB gap, merge
        assert_eq!(
            merge_ranges(&[0..1, 1024 * 1024 + 1..1024 * 1024 + 2]).collect::<Vec<_>>(),
            [(0..1048578, 2)]
        );

        // > 1MiB gap, do not merge
        assert_eq!(
            merge_ranges(&[0..1, 1024 * 1024 + 2..1024 * 1024 + 3]).collect::<Vec<_>>(),
            [(0..1, 1), (1048578..1048579, 2)]
        );

        // <= 12.5% gap, merge
        assert_eq!(
            merge_ranges(&[0..8, 10..11]).collect::<Vec<_>>(),
            [(0..11, 2)]
        );

        // <= 12.5% gap relative to RHS, merge
        assert_eq!(
            merge_ranges(&[0..1, 3..11]).collect::<Vec<_>>(),
            [(0..11, 2)]
        );

        // Overlapping range, merge
        assert_eq!(
            merge_ranges(&[0..80 * 1024 * 1024, 10 * 1024 * 1024..70 * 1024 * 1024])
                .collect::<Vec<_>>(),
            [(0..80 * 1024 * 1024, 2)]
        );
    }
}
