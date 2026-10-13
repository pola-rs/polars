use std::sync::Arc;

use polars_async::primitives::wait_group::{WaitGroup, WaitToken};
use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::metrics::OptIOMetrics;
use polars_io::utils::byte_source::{ByteSource, DynByteSource, DynByteSourceBuilder};
use polars_io::utils::compression::SupportedCompression;
use polars_plan::dsl::ScanSource;

/// Number of bytes fetched from the start of a cloud file during initialization. Files up to this
/// size are fetched entirely by a single request, larger files continue reading after it.
///
/// The object size is unknown upfront, so each request reserves this many bytes of the in-flight
/// byte budget. With default settings, `CLOUD_PRE_INIT_LIMIT` readers initializing concurrently
/// reserve 32 MiB, which is the minimum initial byte budget.
const CLOUD_INIT_FETCH_SIZE: usize = 256 * 1024;

/// Preferred number of cloud readers to initialize concurrently. Initialization is a single
/// request of at most `CLOUD_INIT_FETCH_SIZE` bytes in the common case, so this is latency-bound.
pub(crate) const CLOUD_PRE_INIT_LIMIT: usize = 128;

/// State of a line-based file (CSV, NDJSON) after initialization.
#[derive(Clone)]
pub struct InitializedState {
    pub(crate) file_size: usize,
    pub(crate) compression: Option<SupportedCompression>,
    pub(crate) byte_source: Arc<DynByteSource>,
    /// Start of a cloud file fetched during initialization, if the file is larger than it.
    pub(crate) prefix: Option<Buffer<u8>>,
}

impl InitializedState {
    pub(crate) async fn initialize(
        scan_source: ScanSource,
        byte_source_builder: DynByteSourceBuilder,
        cloud_options: Option<Arc<CloudOptions>>,
        io_metrics: OptIOMetrics,
    ) -> PolarsResult<Self> {
        let byte_source = ASYNC
            .spawn(async move {
                scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(
                        &byte_source_builder,
                        cloud_options.as_deref(),
                        io_metrics.0,
                    )
                    .await
            })
            .await
            .unwrap()?;
        let byte_source = Arc::new(byte_source);

        // Fetch the file size and the compression magic bytes in a single request. For cloud
        // sources the request also covers small files entirely, so that these are read from
        // memory without further requests. Scans over many small files are latency-bound.
        // @TODO: Refactor FileInfo so we can re-use the file_size value from the planning stage.
        let is_cloud = byte_source.is_cloud();
        let prefix_len = if is_cloud { CLOUD_INIT_FETCH_SIZE } else { 4 };
        let (prefix, file_size) = {
            // A single small request per file is latency-bound. It is admitted by the bytes-based
            // controller, which allows many concurrent requests, while data reads keep the
            // configured strategy.
            let prefix_source = byte_source
                .with_fetch_config(FetchConfig::random_access())
                .map_or_else(|| byte_source.clone(), Arc::new);
            ASYNC
                .spawn(async move { prefix_source.get_prefix(prefix_len).await })
                .await
                .unwrap()?
        };

        let compression = SupportedCompression::check(&prefix);

        let (byte_source, prefix) = if !is_cloud {
            (byte_source, None)
        } else if prefix.len() == file_size {
            (Arc::new(DynByteSource::from(prefix)), None)
        } else {
            (byte_source, Some(prefix))
        };

        Ok(Self {
            file_size,
            compression,
            byte_source,
            prefix,
        })
    }

    /// Whether the file is read through the async prefetch pipeline. Uncompressed in-memory data
    /// (memory-mapped local files, or small cloud files fetched during initialization) is read
    /// directly.
    pub(crate) fn uses_async_prefetch(&self) -> bool {
        !(matches!(self.byte_source.as_ref(), DynByteSource::Buffer(_))
            && self.compression.is_none())
    }
}

pub struct ChunkPrefetchSync {
    pub(crate) prefetch_limit: usize,
    pub(crate) prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    pub(crate) shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,

    /// Waits for the previous reader to finish spawning prefetches.
    pub(crate) prev_all_spawned: Option<WaitGroup>,
    /// Dropped once the current reader has finished spawning prefetches.
    pub(crate) current_all_spawned: Option<WaitToken>,
}

impl ChunkPrefetchSync {
    /// Takes this reader's place in the prefetch order, called from `prepare_read`.
    ///
    /// Prefetches of consecutive readers are spawned in order. A reader without prefetches stays
    /// out of that order, otherwise the next reader could start before the previous one finished
    /// spawning its prefetches.
    pub(crate) fn prepare_read(&mut self, init_data: &InitializedState) {
        if !init_data.uses_async_prefetch() {
            return;
        }

        let wait_group_this_reader = WaitGroup::default();
        let prefetch_all_spawned_token = wait_group_this_reader.token();

        let prev_wait_group: Option<WaitGroup> = self
            .shared_prefetch_wait_group_slot
            .try_lock()
            .unwrap()
            .replace(wait_group_this_reader);

        self.prev_all_spawned = prev_wait_group;
        self.current_all_spawned = Some(prefetch_all_spawned_token);
    }
}
