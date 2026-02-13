use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::metrics::IOMetrics;
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_plan::dsl::ScanSource;
use polars_utils::relaxed_cell::RelaxedCell;

use crate::async_primitives::wait_group::WaitGroup;
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::ndjson::NDJsonFileReader;
use crate::nodes::io_sources::ndjson::builder::ndjson_reader_capabilities;
use crate::nodes::io_sources::ndjson::chunk_reader::ChunkReaderBuilder;

pub struct LineReaderBuilder {
    pub prefetch_limit: RelaxedCell<usize>,
    pub prefetch_semaphore: std::sync::OnceLock<Arc<tokio::sync::Semaphore>>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for LineReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LineReaderBuilder")
            .field("prefetch_limit", &self.prefetch_limit)
            .field("prefetch_semaphore", &self.prefetch_semaphore)
            .finish()
    }
}

impl FileReaderBuilder for LineReaderBuilder {
    fn reader_name(&self) -> &str {
        "line"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        ndjson_reader_capabilities()
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        // The maximum number of chunks actively being prefetched at any point in time.
        let prefetch_limit = std::env::var("POLARS_LINES_CHUNK_PREFETCH_LIMIT")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .ok()
                    .unwrap_or_else(|| {
                        panic!("invalid value for POLARS_LINES_CHUNK_PREFETCH_LIMIT: {x}")
                    })
                    .get()
            })
            .unwrap_or(execution_state.num_pipelines.saturating_mul(2))
            .max(1);

        self.prefetch_limit.store(prefetch_limit);

        if config::verbose() {
            eprintln!(
                "[LineReaderBuilder]: prefetch_limit: {}",
                self.prefetch_limit.load()
            );
        }

        self.prefetch_semaphore
            .set(Arc::new(tokio::sync::Semaphore::new(prefetch_limit)))
            .unwrap()
    }

    fn set_io_metrics(&self, io_metrics: Arc<IOMetrics>) {
        self.io_metrics.set(io_metrics).ok().unwrap()
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        use crate::metrics::OptIOMetrics;
        use crate::nodes::io_sources::ndjson::ChunkPrefetchSync;

        let scan_source = source;
        let chunk_reader_builder = ChunkReaderBuilder::Lines;
        let verbose = config::verbose();

        let byte_source_builder =
            if scan_source.is_cloud_url() || polars_config::config().force_async() {
                DynByteSourceBuilder::ObjectStore
            } else {
                DynByteSourceBuilder::Mmap
            };

        // Leverage the existing NDJson code path and line counting functionality.
        let reader = NDJsonFileReader {
            scan_source,
            cloud_options,
            chunk_reader_builder,
            count_rows_fn: polars_io::scan_lines::count_lines,
            verbose,
            byte_source_builder,
            chunk_prefetch_sync: ChunkPrefetchSync {
                prefetch_limit: self.prefetch_limit.load(),
                prefetch_semaphore: Arc::clone(self.prefetch_semaphore.get().unwrap()),
                shared_prefetch_wait_group_slot: Arc::clone(&self.shared_prefetch_wait_group_slot),
                prev_all_spawned: None,
                current_all_spawned: None,
            },
            init_data: None,
            io_metrics: OptIOMetrics(self.io_metrics.get().cloned()),
        };

        Box::new(reader) as _
    }
}
