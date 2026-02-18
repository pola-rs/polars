use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::prelude::{FileMetadata, ParallelStrategy, ParquetOptions};
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_plan::dsl::ScanSource;
use polars_utils::relaxed_cell::RelaxedCell;

use super::{FileReader, ParquetFileReader};
use crate::async_primitives::wait_group::WaitGroup;
use crate::metrics::{IOMetrics, OptIOMetrics};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

#[derive(Clone)]
pub struct ParquetReaderBuilder {
    pub first_metadata: Option<Arc<FileMetadata>>,
    pub options: Arc<ParquetOptions>,
    pub prefetch_limit: RelaxedCell<usize>,
    pub prefetch_semaphore: std::sync::OnceLock<Arc<tokio::sync::Semaphore>>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for ParquetReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetReaderBuilder")
            .field("first_metadata", &self.first_metadata)
            .field("options", &self.options)
            .field("prefetch_semaphore", &self.prefetch_semaphore)
            .finish()
    }
}

impl FileReaderBuilder for ParquetReaderBuilder {
    fn reader_name(&self) -> &str {
        "parquet"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        let mut capabilities = RC::ROW_INDEX
            | RC::PRE_SLICE
            | RC::NEGATIVE_PRE_SLICE
            | RC::PARTIAL_FILTER
            | RC::MAPPED_COLUMN_PROJECTION;

        if matches!(
            self.options.parallel,
            ParallelStrategy::Auto | ParallelStrategy::Prefiltered
        ) {
            capabilities |= RC::FULL_FILTER;
        }
        capabilities
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        let prefetch_limit = std::env::var("POLARS_ROW_GROUP_PREFETCH_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .unwrap_or_else(|_| {
                        panic!("invalid value for POLARS_ROW_GROUP_PREFETCH_SIZE: {x}")
                    })
                    .get()
            })
            .unwrap_or(execution_state.num_pipelines.saturating_mul(2))
            .max(1);

        self.prefetch_limit.store(prefetch_limit);

        if config::verbose() {
            eprintln!(
                "[ParquetReaderBuilder]: prefetch_limit: {}",
                self.prefetch_limit.load()
            );
        }

        self.prefetch_semaphore
            .set(Arc::new(tokio::sync::Semaphore::new(prefetch_limit)))
            .unwrap()
    }

    fn set_io_metrics(&self, io_metrics: Arc<IOMetrics>) {
        let _ = self.io_metrics.set(io_metrics);
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        use crate::nodes::io_sources::parquet::RowGroupPrefetchSync;

        let scan_source = source;
        let config = self.options.clone();
        let verbose = config::verbose();

        let byte_source_builder =
            if scan_source.is_cloud_url() || polars_config::config().force_async() {
                DynByteSourceBuilder::ObjectStore
            } else {
                DynByteSourceBuilder::Mmap
            };

        assert!(self.prefetch_limit.load() > 0);

        let reader = ParquetFileReader {
            scan_source,
            cloud_options,
            config,
            metadata: if scan_source_idx == 0 {
                self.first_metadata.clone()
            } else {
                None
            },
            byte_source_builder,
            row_group_prefetch_sync: RowGroupPrefetchSync {
                prefetch_limit: self.prefetch_limit.load(),
                prefetch_semaphore: Arc::clone(self.prefetch_semaphore.get().unwrap()),
                shared_prefetch_wait_group_slot: Arc::clone(&self.shared_prefetch_wait_group_slot),
                prev_all_spawned: None,
                current_all_spawned: None,
            },
            io_metrics: OptIOMetrics(self.io_metrics.get().cloned()),
            verbose,

            init_data: None,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
