use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_async::primitives::wait_group::WaitGroup;
use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::concurrency::get_request_budget;
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::prelude::{FileMetadata, ParallelStrategy, ParquetOptions};
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_plan::dsl::ScanSource;

use super::super::shared::pipeline_budget::PipelineBudget;
use super::{FileReader, ParquetFileReader};
use crate::metrics::{IOMetrics, OptIOMetrics};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

#[derive(Clone)]
pub struct ParquetReaderBuilder {
    pub first_metadata: Option<Arc<FileMetadata>>,
    pub options: Arc<ParquetOptions>,
    pub pipeline_budget: std::sync::OnceLock<PipelineBudget>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for ParquetReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetReaderBuilder")
            .field("first_metadata", &self.first_metadata)
            .field("options", &self.options)
            .field("pipeline_budget", &self.pipeline_budget)
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
        // Bound the number of fetches in the pipeline.
        // This bound goes together with the `prefetch_kbytes_limit` bound. In most
        // large-dataset use cases, the kbytes memory bound will kick in first.
        // This limit should be at least as large as the max in-flight concurrency.
        let prefetch_limit = std::env::var("POLARS_ROW_GROUP_PREFETCH_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .unwrap_or_else(|_| {
                        panic!("invalid value for POLARS_ROW_GROUP_PREFETCH_SIZE: {x}")
                    })
                    .get()
            })
            .unwrap_or(
                //kdn TODO TEST & TUNE
                execution_state
                    .num_pipelines
                    .saturating_mul(2)
                    .max(get_request_budget() as usize)
                    .clamp(16, 2048),
            )
            .max(1);

        // Bound the max memory in use for the pipeline.
        // This should be large enough to be non-blocking, but small enough to avoid
        // excessive memory use from a run-away prefetch pipeline.
        // "Correct" formula: (a) max effective in-flight bdp-based bytes budget + (b) decode pipeline.
        // Since we do not know (a) at startup, we use  '3 * (b) + buffer' as a proxy for (a), where the
        // multiplier reflects the max gain factor for in-flight control.
        // TODO: Dynamically adapt the max memory to the observed BDP.
        // NOTE: This does not account for the decompression multiplier, so actual memory
        // usage can be substantially larger.
        let prefetch_kbytes_limit = std::env::var("POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .unwrap_or_else(|_| {
                        panic!("invalid value for POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET: {x}")
                    })
                    .get()
            })
            .unwrap_or({
                let target_chunk_size_kb = FetchConfig::random_access().chunk_size.div_ceil(1024);
                4 * execution_state.num_pipelines * target_chunk_size_kb
            })
            // Avoid deadlock.
            .max(polars_io::cloud::concurrency_config::get_download_chunk_size().div_ceil(1024));

        if config::verbose() {
            eprintln!(
                "[ParquetReaderBuilder]: prefetch_limit: {}, prefetch_kbytes_limit: {}",
                prefetch_limit, prefetch_kbytes_limit
            );
        }

        self.pipeline_budget
            .set(PipelineBudget::new(prefetch_limit, prefetch_kbytes_limit))
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
                DynByteSourceBuilder::ObjectStore(FetchConfig::random_access())
            } else {
                DynByteSourceBuilder::Mmap
            };

        let pipeline_budget = self
            .pipeline_budget
            .get()
            .expect("set_execution_state must be called before build_file_reader")
            .clone();

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
                pipeline_budget,
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
