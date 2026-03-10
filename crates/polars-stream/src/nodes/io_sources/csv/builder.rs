use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::metrics::IOMetrics;
use polars_io::prelude::CsvReadOptions;
use polars_plan::dsl::ScanSource;
use polars_utils::relaxed_cell::RelaxedCell;

use super::{CsvFileReader, DynByteSourceBuilder};
use crate::async_primitives::wait_group::WaitGroup;
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

pub struct CsvReaderBuilder {
    pub options: Arc<CsvReadOptions>,
    pub prefetch_limit: RelaxedCell<usize>,
    pub prefetch_semaphore: std::sync::OnceLock<Arc<tokio::sync::Semaphore>>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for CsvReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsvReaderBuilder")
            .field("ignore_errors", &self.options.ignore_errors)
            .field("prefetch_limit", &self.prefetch_limit)
            .field("prefetch_semaphore", &self.prefetch_semaphore)
            .finish()
    }
}

impl FileReaderBuilder for CsvReaderBuilder {
    fn reader_name(&self) -> &str {
        "csv"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        if self.options.parse_options.comment_prefix.is_some() {
            RC::empty()
        } else {
            RC::PRE_SLICE
        }
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        // The maximum number of chunks actively being prefetched at any given point in time.
        let prefetch_limit = std::env::var("POLARS_CSV_CHUNK_PREFETCH_LIMIT")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .ok()
                    .unwrap_or_else(|| {
                        panic!("invalid value for POLARS_CSV_CHUNK_PREFETCH_LIMIT: {x}")
                    })
                    .get()
            })
            .unwrap_or(execution_state.num_pipelines.saturating_mul(2))
            .max(1);

        self.prefetch_limit.store(prefetch_limit);

        if config::verbose() {
            eprintln!(
                "[CsvReaderBuilder]: prefetch_limit: {}",
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
        use crate::nodes::io_sources::csv::ChunkPrefetchSync;

        let scan_source = source;
        let verbose = config::verbose();
        let options = self.options.clone();

        let byte_source_builder =
            if scan_source.is_cloud_url() || polars_config::config().force_async() {
                DynByteSourceBuilder::ObjectStore
            } else {
                DynByteSourceBuilder::Mmap
            };

        let reader = CsvFileReader {
            scan_source,
            cloud_options,
            options,
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

        Box::new(reader) as Box<dyn FileReader>
    }
}
