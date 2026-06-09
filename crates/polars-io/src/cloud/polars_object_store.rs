use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use futures::{Stream, StreamExt as _, TryStreamExt as _};
use hashbrown::hash_map::RawEntryMut;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use polars_buffer::Buffer;
use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_error::{PolarsError, PolarsResult};
use polars_utils::pl_path::PlRefPath;
use tokio::io::AsyncWriteExt;

use crate::cloud::concurrency::IoSample;
use crate::pl_async::{
    self, MAX_BUDGET_PER_REQUEST, get_concurrency_limit, get_download_chunk_size,
    get_random_access_chunk_size, get_streaming_chunk_size, tune_with_concurrency_budget,
    with_concurrency_budget,
};

/// Determines how in-flight concurrency for access to the back-end store is handled.
#[derive(Clone, Debug, Copy)]
pub enum ConcurrencyStrategy {
    /// (Almost) no in-flight concurrency control.
    /// Warning: this may result in an unbounded API call rate. Only use in the
    /// context of a rate-limited pipeline.
    Unbounded,
    /// In-flight concurrency control using a semi-static count-based budget.
    /// NOTE: This is a legacy strategy which does not scale up to the full potential.
    Legacy,
    /// In-flight concurrency control using a dynamically sensed bytes-budget, backed by
    /// a count-budget as fallback.
    BytesBased,
}

#[derive(Clone, Copy, Debug)]
pub struct FetchConfig {
    pub chunk_size: usize,
    pub strategy: ConcurrencyStrategy,
}

impl FetchConfig {
    /// Use for file formats that are randomly accessible, i.e. individual
    /// row groups (or record batches) and/or individual columns can be fetched
    /// directly using the metadata as an input. Example: Parquet, IPC with internal
    /// extensions.
    ///
    /// The chunk_size should be smaller to enable smooth operation of the
    /// bytes-based in-flight concurrency controller.
    pub fn random_access() -> Self {
        Self {
            chunk_size: get_random_access_chunk_size(),
            strategy: ConcurrencyStrategy::BytesBased,
        }
    }

    /// Use for file formats that have a sequential layout, i.e. the file bytes
    /// must be fetched and parsed sequentially. The pipeline is responsible for
    /// managing back-pressure and rate-limiting. Example: CSV.

    pub fn streaming() -> Self {
        Self {
            chunk_size: get_streaming_chunk_size(),
            // TODO: For now - keep as Legacy. Switch to Unbounded in a future PR.
            strategy: ConcurrencyStrategy::Legacy,
        }
    }

    /// Used for legacy fetch.
    /// @TOOD: Deprecate over time.
    pub fn legacy() -> Self {
        Self {
            chunk_size: get_download_chunk_size(),
            strategy: ConcurrencyStrategy::Legacy,
        }
    }
}

#[derive(Debug)]
pub struct PolarsObjectStoreError {
    pub base_url: PlRefPath,
    pub source: object_store::Error,
}

impl PolarsObjectStoreError {
    pub fn from_url(base_url: &PlRefPath) -> impl FnOnce(object_store::Error) -> Self {
        |error| Self {
            base_url: base_url.clone(),
            source: error,
        }
    }
}

impl Display for PolarsObjectStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "object-store error: {} (path: {})",
            self.source, &self.base_url
        )
    }
}

impl std::error::Error for PolarsObjectStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl From<PolarsObjectStoreError> for std::io::Error {
    fn from(value: PolarsObjectStoreError) -> Self {
        std::io::Error::other(value)
    }
}

impl From<PolarsObjectStoreError> for PolarsError {
    fn from(value: PolarsObjectStoreError) -> Self {
        PolarsError::IO {
            error: Arc::new(value.into()),
            msg: None,
        }
    }
}

mod inner {

    use std::borrow::Cow;
    use std::future::Future;
    use std::sync::Arc;

    use object_store::ObjectStore;
    use polars_core::config;
    use polars_error::{PolarsError, PolarsResult};
    use polars_utils::relaxed_cell::RelaxedCell;

    use crate::cloud::concurrency::{ConcurrencyController, ControllerConfig};
    use crate::cloud::{ObjectStoreErrorContext, PolarsObjectStoreBuilder};
    use crate::metrics::{IOMetrics, OptIOMetrics};

    #[derive(Debug)]
    struct Inner {
        store: tokio::sync::RwLock<Arc<dyn ObjectStore>>,
        builder: PolarsObjectStoreBuilder,
        rebuilt: RelaxedCell<bool>,
    }

    /// Polars wrapper around [`ObjectStore`] functionality. This struct is cheaply cloneable.
    #[derive(Clone, Debug)]
    pub struct PolarsObjectStore {
        inner: Arc<Inner>,
        /// Avoid contending the Mutex `lock()` until the first re-build.
        initial_store: std::sync::Arc<dyn ObjectStore>,
        io_metrics: OptIOMetrics,
        /// In-flight concurrency control using the (new) BDP model.
        concurrency: Arc<std::sync::OnceLock<Arc<ConcurrencyController>>>,
    }

    impl PolarsObjectStore {
        pub(crate) fn new_from_inner(
            store: Arc<dyn ObjectStore>,
            builder: PolarsObjectStoreBuilder,
        ) -> Self {
            let initial_store = store.clone();
            Self {
                inner: Arc::new(Inner {
                    store: tokio::sync::RwLock::new(store),
                    builder,
                    rebuilt: RelaxedCell::from(false),
                }),
                initial_store,
                io_metrics: OptIOMetrics(None),
                concurrency: Arc::new(std::sync::OnceLock::new()), // Arc::new(ConcurrencyController::new(ControllerConfig::default())),
            }
        }

        pub fn set_io_metrics(&mut self, io_metrics: Option<Arc<IOMetrics>>) -> &mut Self {
            self.io_metrics = OptIOMetrics(io_metrics);
            self
        }

        pub fn io_metrics(&self) -> &OptIOMetrics {
            &self.io_metrics
        }

        pub fn get_or_init_concurrency(&self) -> &Arc<ConcurrencyController> {
            self.concurrency
                .get_or_init(|| Arc::new(ConcurrencyController::new(ControllerConfig::default())))
        }

        /// Gets the underlying [`ObjectStore`] implementation.
        pub async fn to_dyn_object_store(&self) -> Cow<'_, Arc<dyn ObjectStore>> {
            if !self.inner.rebuilt.load() {
                Cow::Borrowed(&self.initial_store)
            } else {
                Cow::Owned(self.inner.store.read().await.clone())
            }
        }

        pub async fn rebuild_inner(
            &self,
            from_version: &Arc<dyn ObjectStore>,
        ) -> PolarsResult<Arc<dyn ObjectStore>> {
            let mut current_store = self.inner.store.write().await;

            // If this does not eq, then `inner` was already re-built by another thread.
            if Arc::ptr_eq(&*current_store, from_version) {
                *current_store =
                    self.inner
                        .builder
                        .clone()
                        .build_impl(true)
                        .await
                        .map_err(|e| {
                            e.wrap_msg(|e| format!("attempt to rebuild object store failed: {e}"))
                        })?;
            }

            self.inner.rebuilt.store(true);

            Ok((*current_store).clone())
        }

        pub async fn exec_with_rebuild_retry_on_err<'s, 'f, Fn, Fut, O>(
            &'s self,
            mut func: Fn,
        ) -> PolarsResult<O>
        where
            Fn: FnMut(Cow<'s, Arc<dyn ObjectStore>>) -> Fut + 'f,
            Fut: Future<Output = object_store::Result<O>>,
        {
            let store = self.to_dyn_object_store().await;

            let out = func(store.clone()).await;

            let orig_err = match out {
                Ok(v) => return Ok(v),
                Err(e) => e,
            };

            if config::verbose() {
                eprintln!(
                    "[PolarsObjectStore]: got error: {}, will rebuild store and retry",
                    &orig_err
                );
            }

            let store = self
                .rebuild_inner(&store)
                .await
                .map_err(|e| e.wrap_msg(|e| format!("{e}; original error: {orig_err}")))?;

            func(Cow::Owned(store)).await.map_err(|e| {
                let e: PolarsError = self.error_context().attach_err_info(e).into();

                if self.inner.builder.is_azure()
                    && std::env::var("POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY").as_deref()
                        != Ok("1")
                {
                    // Note: This error is intended for Python audiences. The logic for retrieving
                    // these keys exist only on the Python side.
                    e.wrap_msg(|e| {
                        format!(
                            "{e}; note: if you are using Python, consider setting \
POLARS_AUTO_USE_AZURE_STORAGE_ACCOUNT_KEY=1 if you would like polars to try to retrieve \
and use the storage account keys from Azure CLI to authenticate"
                        )
                    })
                } else {
                    e
                }
            })
        }

        pub fn error_context(&self) -> ObjectStoreErrorContext {
            ObjectStoreErrorContext::new(self.inner.builder.path().clone())
        }
    }
}

#[derive(Clone)]
pub struct ObjectStoreErrorContext {
    path: PlRefPath,
}

impl ObjectStoreErrorContext {
    pub fn new(path: PlRefPath) -> Self {
        Self { path }
    }

    pub fn attach_err_info(self, err: object_store::Error) -> PolarsObjectStoreError {
        let ObjectStoreErrorContext { path } = self;

        PolarsObjectStoreError {
            base_url: path,
            source: err,
        }
    }
}

pub use inner::PolarsObjectStore;

pub type ObjectStorePath = object_store::path::Path;

impl PolarsObjectStore {
    pub fn build_buffered_ranges_stream<'a, T: Iterator<Item = Range<usize>>>(
        &'a self,
        path: &'a Path,
        ranges: T,
        strategy: ConcurrencyStrategy,
    ) -> impl Stream<Item = PolarsResult<Buffer<u8>>> + use<'a, T> {
        let controller = match strategy {
            ConcurrencyStrategy::BytesBased => Some(self.get_or_init_concurrency().clone()),
            ConcurrencyStrategy::Unbounded | ConcurrencyStrategy::Legacy => None,
        };

        let n_buffered = match strategy {
            // In case of bytes-based concurrency, the concurrency is controlled by the
            // admission semaphore in the pipeline.
            // The buffered size is set to a large constant as a backstop. Once all
            // callsites are verified to pass finite, metadata-derived ranges, this can be
            // set to usize::MAX.
            ConcurrencyStrategy::BytesBased => 4096,
            ConcurrencyStrategy::Unbounded | ConcurrencyStrategy::Legacy => {
                get_concurrency_limit() as usize
            },
        };

        futures::stream::iter(ranges.map(move |range| {
            let controller = controller.clone();
            async move {
                if range.is_empty() {
                    return Ok(Buffer::new());
                }
                let bytes_req = range.len() as u64;

                // Held until end of block to bound in-flight bytes.
                let _permit = match &controller {
                    Some(controller) => Some(controller.acquire(bytes_req).await),
                    None => None,
                };

                // kdn TODO REFACTOR - 3 things happening at once
                let (out, ttfb) = self
                    .io_metrics()
                    .record_io_read(
                        bytes_req,
                        self.exec_with_rebuild_retry_on_err(|s| async move {
                            let t0 = Instant::now();
                            let response = s
                                .get_opts(
                                    path,
                                    object_store::GetOptions {
                                        range: Some((range.start as u64..range.end as u64).into()),
                                        ..Default::default()
                                    },
                                )
                                .await?;
                            let t1 = t0.elapsed();
                            let out = response.bytes().await?;

                            Ok((out, t1))
                        }),
                    )
                    .await?;

                if let Some(controller) = &controller {
                    controller.record_io(IoSample {
                        n_bytes: out.len() as u64,
                        ttfb,
                        completion_time: Instant::now(),
                    });
                }

                Ok(Buffer::from_owner(out))
            }
        }))
        .buffered(n_buffered)
    }

    pub async fn get_range(
        &self,
        path: &Path,
        range: Range<usize>,
        config: FetchConfig,
    ) -> PolarsResult<Buffer<u8>> {
        if range.is_empty() {
            return Ok(Buffer::new());
        }

        let parts = split_range(range.clone(), Some(config.chunk_size));

        match config.strategy {
            ConcurrencyStrategy::Legacy => self.get_range_legacy(path, range).await,
            ConcurrencyStrategy::Unbounded | ConcurrencyStrategy::BytesBased => self
                .build_buffered_ranges_stream(path, parts, config.strategy)
                .try_collect::<Vec<_>>()
                .await
                .map(|parts| {
                    if parts.len() == 1 {
                        return parts.into_iter().next().unwrap();
                    }
                    let mut combined = Vec::with_capacity(range.len());
                    for part in parts {
                        combined.extend_from_slice(&part);
                    }
                    assert_eq!(combined.len(), range.len());
                    Buffer::from_vec(combined)
                }),
        }
    }

    async fn get_range_legacy(&self, path: &Path, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        if range.is_empty() {
            return Ok(Buffer::new());
        }

        let parts = split_range(range.clone(), None);

        if parts.len() == 1 {
            let out = tune_with_concurrency_budget(1, move || async move {
                let bytes = self
                    .io_metrics()
                    .record_io_read(
                        range.len() as u64,
                        self.exec_with_rebuild_retry_on_err(|s| async move {
                            s.get_range(path, range.start as u64..range.end as u64)
                                .await
                        }),
                    )
                    .await?;

                PolarsResult::Ok(Buffer::from_owner(bytes))
            })
            .await?;

            Ok(out)
        } else {
            let parts = tune_with_concurrency_budget(
                parts.len().clamp(0, MAX_BUDGET_PER_REQUEST) as u32,
                || {
                    self.build_buffered_ranges_stream(path, parts, ConcurrencyStrategy::Legacy)
                        .try_collect::<Vec<Buffer<u8>>>()
                },
            )
            .await?;

            let mut combined = Vec::with_capacity(range.len());

            for part in parts {
                combined.extend_from_slice(&part)
            }

            assert_eq!(combined.len(), range.len());

            PolarsResult::Ok(Buffer::from_vec(combined))
        }
    }

    pub async fn get_ranges_sort(
        &self,
        path: &Path,
        ranges: &mut [Range<usize>],
        config: FetchConfig,
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        if ranges.is_empty() {
            return Ok(Default::default());
        }

        ranges.sort_unstable_by_key(|x| x.start);

        let ranges_len = ranges.len();
        let (merged_ranges, merged_ends): (Vec<_>, Vec<_>) =
            merge_ranges(ranges, Some(config.chunk_size)).unzip();

        let mut out = PlHashMap::with_capacity(ranges_len);

        // Build an inflight admission-aware stream over the merged ranges.
        let mut stream =
            self.build_buffered_ranges_stream(path, merged_ranges.iter().cloned(), config.strategy);

        let mut current_offset = 0;
        let mut ends_iter = merged_ends.iter();
        let mut splitted_parts: Vec<Buffer<u8>> = vec![];

        while let Some(bytes) = stream.try_next().await? {
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
                Buffer::from(out)
            };

            assert_eq!(bytes.len(), full_range.len());

            for range in &ranges[current_offset..end] {
                let slice = bytes
                    .clone()
                    .sliced(range.start - full_range.start..range.end - full_range.start);

                match out.raw_entry_mut().from_key(&range.start) {
                    RawEntryMut::Vacant(slot) => {
                        slot.insert(range.start, slice);
                    },
                    RawEntryMut::Occupied(mut slot) => {
                        if slot.get_mut().len() < slice.len() {
                            *slot.get_mut() = slice;
                        }
                    },
                }
            }

            current_offset = end;
        }

        assert!(splitted_parts.is_empty());

        Ok(out)
    }

    // kdn TODO REFACTOR: review and update concurrency strategy
    pub async fn download(&self, path: &Path, file: &mut tokio::fs::File) -> PolarsResult<()> {
        let size = self.head(path, ConcurrencyStrategy::Unbounded).await?.size;
        let parts = split_range(0..size as usize, None);

        // TODO: Replace the legacy concurrency_budget call and switch to BytesBased inflight
        // admission control.
        tune_with_concurrency_budget(
            parts.len().clamp(0, MAX_BUDGET_PER_REQUEST) as u32,
            || async {
                let mut stream =
                    self.build_buffered_ranges_stream(path, parts, ConcurrencyStrategy::Unbounded);
                let mut len = 0;
                while let Some(bytes) = stream.try_next().await? {
                    len += bytes.len();
                    file.write_all(&bytes).await?;
                }

                assert_eq!(len, size as usize);

                PolarsResult::Ok(pl_async::Size::from(len as u64))
            },
        )
        .await?;

        // Dropping is delayed for tokio async files so we need to explicitly
        // flush here (https://github.com/tokio-rs/tokio/issues/2307#issuecomment-596336451).
        file.sync_all().await.map_err(PolarsError::from)?;

        Ok(())
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    pub async fn head(
        &self,
        path: &Path,
        strategy: ConcurrencyStrategy,
    ) -> PolarsResult<ObjectMeta> {
        //kdn TODO REFACTOR: Update concurrency strategy.
        // For now, we fall back to 'Legacy' which is fine for metadata.
        // Since this carries an early signal, the IO Sample is of interest regardless of
        // the strategy in use.
        with_concurrency_budget(1, || {
            self.exec_with_rebuild_retry_on_err(|s| {
                async move {
                    let t0 = Instant::now();
                    let head_result = self.io_metrics().record_io_read(0, s.head(path)).await;
                    if let ConcurrencyStrategy::BytesBased = strategy {
                        // self.get_or_init_concurrency().record_ttfb(ttfb);
                        self.get_or_init_concurrency().record_io(IoSample {
                            n_bytes: 0,
                            ttfb: t0.elapsed(),
                            completion_time: Instant::now(),
                        });
                    }

                    if head_result.is_err() {
                        let t0 = Instant::now();
                        // Pre-signed URLs forbid the HEAD method, but we can still retrieve the header
                        // information with a range 0-1 request.
                        let get_range_0_1_result = self
                            .io_metrics()
                            .record_io_read(
                                0,
                                s.get_opts(
                                    path,
                                    object_store::GetOptions {
                                        range: Some((0..1).into()),
                                        ..Default::default()
                                    },
                                ),
                            )
                            .await;

                        if let ConcurrencyStrategy::BytesBased = strategy {
                            self.get_or_init_concurrency().record_io(IoSample {
                                n_bytes: 0,
                                ttfb: t0.elapsed(),
                                completion_time: Instant::now(),
                            });
                        }

                        if let Ok(v) = get_range_0_1_result {
                            return Ok(v.meta);
                        }
                    }

                    let out = head_result?;

                    Ok(out)
                }
            })
        })
        .await
    }
}

/// Splits a single range into multiple smaller ranges, which can be downloaded concurrently for
/// much higher throughput.
fn split_range(
    range: Range<usize>,
    chunk_size: Option<usize>,
) -> impl ExactSizeIterator<Item = Range<usize>> {
    let chunk_size = chunk_size.unwrap_or_else(|| get_download_chunk_size());

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
fn merge_ranges(
    ranges: &[Range<usize>],
    chunk_size: Option<usize>,
) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
    let chunk_size = chunk_size.unwrap_or_else(|| get_download_chunk_size());

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
        .flat_map(move |x| {
            // Split large individual ranges within the list of ranges.
            let (range, end) = x;
            let split = split_range(range, Some(chunk_size));
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
            assert_eq!(split_range(0..0, None).collect::<Vec<_>>(), [0..0]);
            assert_eq!(split_range(3..3, None).collect::<Vec<_>>(), [3..3]);
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
            assert_eq!(split_range(0..n, None).collect::<Vec<_>>(), [0..89478485]);
        }

        assert_eq!(
            split_range(0..n + 1, None).collect::<Vec<_>>(),
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
            split_range(0..n, None).collect::<Vec<_>>(),
            [0..80530637, 80530637..161061273]
        );

        assert_eq!(
            split_range(0..n + 1, None).collect::<Vec<_>>(),
            [0..53687092, 53687092..107374183, 107374183..161061274]
        );
    }

    #[test]
    fn test_merge_ranges() {
        use super::{get_download_chunk_size, merge_ranges};

        let chunk_size = get_download_chunk_size();

        assert_eq!(chunk_size, 64 * 1024 * 1024);

        // Round-trip empty slice
        assert_eq!(merge_ranges(&[], None).collect::<Vec<_>>(), []);

        // We have 1 tiny request followed by 1 huge request. They are combined as it reduces the
        // `abs_diff()` to the `chunk_size`, but afterwards they are split to 2 evenly sized
        // requests.
        assert_eq!(
            merge_ranges(&[0..1, 1..127 * 1024 * 1024], None).collect::<Vec<_>>(),
            [(0..66584576, 0), (66584576..133169152, 2)]
        );

        // <= 1MiB gap, merge
        assert_eq!(
            merge_ranges(&[0..1, 1024 * 1024 + 1..1024 * 1024 + 2], None).collect::<Vec<_>>(),
            [(0..1048578, 2)]
        );

        // > 1MiB gap, do not merge
        assert_eq!(
            merge_ranges(&[0..1, 1024 * 1024 + 2..1024 * 1024 + 3], None).collect::<Vec<_>>(),
            [(0..1, 1), (1048578..1048579, 2)]
        );

        // <= 12.5% gap, merge
        assert_eq!(
            merge_ranges(&[0..8, 10..11], None).collect::<Vec<_>>(),
            [(0..11, 2)]
        );

        // <= 12.5% gap relative to RHS, merge
        assert_eq!(
            merge_ranges(&[0..1, 3..11], None).collect::<Vec<_>>(),
            [(0..11, 2)]
        );

        // Overlapping range, merge
        assert_eq!(
            merge_ranges(
                &[0..80 * 1024 * 1024, 10 * 1024 * 1024..70 * 1024 * 1024],
                None
            )
            .collect::<Vec<_>>(),
            [(0..80 * 1024 * 1024, 2)]
        );
    }
}
