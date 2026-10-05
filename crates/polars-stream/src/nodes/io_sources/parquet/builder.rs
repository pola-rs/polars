use std::sync::Arc;

use polars_async::executor::TaskMetricAggregator;
use polars_async::primitives::wait_group::WaitGroup;
use polars_buffer::Buffer;
use polars_core::config;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::concurrency::get_inflight_request_budget;
use polars_io::cloud::concurrency_config::FetchConfig;
use polars_io::prelude::{FileMetadata, ParallelStrategy, ParquetOptions};
use polars_io::utils::byte_source::{self, DynByteSourceBuilder, FileReadContext};
use polars_plan::dsl::ScanSource;
use polars_utils::pl_str::PlSmallStr;

use super::super::shared::pipeline_budget::{
    PipelineBudget, default_prefetch_kbytes_limit, env_nonzero_usize,
};
use super::{FileReader, ParquetFileReader};
use crate::metrics::{IOMetrics, OptIOMetrics};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

#[derive(Clone)]
pub struct ParquetReaderBuilder {
    pub first_metadata: Option<Arc<FileMetadata>>,
    pub bytes_per_source: Option<Buffer<u64>>,
    pub options: Arc<ParquetOptions>,
    pub pipeline_budget: std::sync::OnceLock<PipelineBudget>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    /// Shared with every file in the scan. Only relevant for `DynByteSourceBuilder::FilePread`.
    pub file_read_context: std::sync::OnceLock<FileReadContext>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
    pub task_metrics: std::sync::OnceLock<Arc<TaskMetricAggregator>>,
}

impl std::fmt::Debug for ParquetReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetReaderBuilder")
            .field("first_metadata", &self.first_metadata)
            .field("bytes_per_source", &self.bytes_per_source)
            .field("options", &self.options)
            .field("pipeline_budget", &self.pipeline_budget)
            .field("read_context", &self.file_read_context)
            .finish()
    }
}

impl FileReaderBuilder for ParquetReaderBuilder {
    fn reader_name(&self) -> PolarsResult<PlSmallStr> {
        Ok(PlSmallStr::from_static("parquet"))
    }

    fn reader_capabilities(&self) -> PolarsResult<ReaderCapabilities> {
        use ReaderCapabilities as RC;

        let mut capabilities = RC::ROW_INDEX
            | RC::PRE_SLICE
            | RC::NEGATIVE_PRE_SLICE
            | RC::PARTIAL_FILTER
            | RC::MAPPED_COLUMN_PROJECTION
            | RC::UNORDERED_FILES;

        if matches!(
            self.options.parallel,
            ParallelStrategy::Auto | ParallelStrategy::Prefiltered
        ) {
            capabilities |= RC::FULL_FILTER;
        }

        Ok(capabilities)
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        if let Some(task_metrics) = execution_state.task_metrics.clone() {
            let _ = self.task_metrics.set(task_metrics);
        }

        let prefetch_limit_env = env_nonzero_usize("POLARS_ROW_GROUP_PREFETCH_SIZE");

        // Bound the number of fetches in the pipeline.
        // This bound goes together with the `prefetch_kbytes_limit` bound; it binds first for
        // row groups smaller than `prefetch_kbytes_limit / prefetch_limit`.
        // This limit should be at least as large as the max in-flight concurrency.
        let prefetch_limit = prefetch_limit_env.unwrap_or_else(|| {
            execution_state
                .num_pipelines
                .saturating_mul(2)
                .max(get_inflight_request_budget() as usize)
                .clamp(16, PipelineBudget::MAX_DEFAULT_COUNT_LIMIT)
        });

        let prefetch_kbytes_limit = env_nonzero_usize("POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET")
            .unwrap_or_else(|| default_prefetch_kbytes_limit(execution_state.num_pipelines));

        if config::verbose() {
            eprintln!(
                "[ParquetReaderBuilder]: prefetch_limit: {}, prefetch_kbytes_limit: {}",
                prefetch_limit, prefetch_kbytes_limit
            );
        }

        let can_grow_for_ordered = prefetch_limit_env.is_none();

        self.pipeline_budget
            .set(PipelineBudget::new(
                prefetch_limit,
                prefetch_kbytes_limit,
                can_grow_for_ordered,
            ))
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
    ) -> PolarsResult<Box<dyn FileReader>> {
        use crate::nodes::io_sources::parquet::RowGroupPrefetchSync;

        let scan_source = source;
        let config = self.options.clone();
        let verbose = config::verbose();

        let byte_source_builder =
            if scan_source.is_cloud_url() || polars_config::config().force_async() {
                DynByteSourceBuilder::ObjectStore(FetchConfig::random_access())
            } else if scan_source.is_buffer() {
                DynByteSourceBuilder::Mmap
            } else {
                let read_context = self.file_read_context.get_or_init(|| {
                    let cfg = polars_config::config();
                    let enable_o_direct = cfg.direct_io();
                    let concurrency = cfg.file_read_concurrency().max(1) as usize;

                    // TODO: Posix_fadv should follow the access-pattern, which varies
                    // by file type and projection.
                    let fadv = cfg.file_posix_fadv();

                    if config::verbose() {
                        eprintln!(
                            "[ParquetReaderBuilder]: file read_context as configured: \
                                read_concurrency: {concurrency}, \
                                posix_fadv: {fadv}, \
                                o_direct: {enable_o_direct}"
                        );
                    }

                    FileReadContext {
                        enable_o_direct,
                        concurrency,
                        permits: byte_source::global_read_permits(),
                        advice: fadv,
                    }
                });
                DynByteSourceBuilder::FilePread(read_context.clone())
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
            file_size: self
                .bytes_per_source
                .as_ref()
                .and_then(|sizes| usize::try_from(sizes[scan_source_idx]).ok()),
            byte_source_builder,
            row_group_prefetch_sync: RowGroupPrefetchSync {
                pipeline_budget,
                shared_prefetch_wait_group_slot: Arc::clone(&self.shared_prefetch_wait_group_slot),
                prev_all_spawned: None,
                current_all_spawned: None,
            },
            io_metrics: OptIOMetrics(self.io_metrics.get().cloned()),
            task_metrics: self.task_metrics.get().cloned(),
            verbose,

            init_data: None,
        };

        Ok(Box::new(reader) as Box<dyn FileReader>)
    }
}
