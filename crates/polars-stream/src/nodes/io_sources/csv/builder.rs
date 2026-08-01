use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_async::primitives::wait_group::WaitGroup;
use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::concurrency_config::FetchConfig;
#[cfg(feature = "csv")]
use polars_io::metrics::IOMetrics;
use polars_io::prelude::CsvReadOptions;
use polars_io::utils::compression::ByteSourceReader;
use polars_io::utils::stream_buf_reader::ReaderSource;
use polars_plan::dsl::ScanSource;
use polars_utils::relaxed_cell::RelaxedCell;

use super::{CsvFileReader, DynByteSourceBuilder, LINE_BATCH_DISTRIBUTOR_BUFFER_SIZE};
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::shared::pipeline_budget::PipelineBudget;

pub struct CsvReaderBuilder {
    pub options: Arc<CsvReadOptions>,
    pub prefetch_limit: RelaxedCell<usize>,
    pub prefetch_semaphore: std::sync::OnceLock<Arc<tokio::sync::Semaphore>>,
    pub line_batch_budget: std::sync::OnceLock<PipelineBudget>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for CsvReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsvReaderBuilder")
            .field("ignore_errors", &self.options.ignore_errors)
            .field("prefetch_limit", &self.prefetch_limit)
            .field("prefetch_semaphore", &self.prefetch_semaphore)
            .field("line_batch_budget", &self.line_batch_budget)
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

        // Bound the mmap-backed or decompressed line batches waiting for CSV
        // decoders. The permit lives through decoding, so the default covers
        // one decoding and one queued batch per pipeline, plus the batch
        // prepared by the producer before it acquires a permit. This keeps the
        // budget out of the ordinary distributor's progress path while still
        // bounding unusually wide or decompressed batches.
        let ordinary_line_batch_limit = execution_state
            .num_pipelines
            .saturating_mul(LINE_BATCH_DISTRIBUTOR_BUFFER_SIZE.saturating_add(1))
            .saturating_add(1)
            .max(2);
        let line_batch_count_limit = ordinary_line_batch_limit;
        let ideal_line_batch_kbytes =
            ByteSourceReader::<ReaderSource>::ideal_read_size().div_ceil(1 << 10);
        let line_batch_kbytes_limit =
            ordinary_line_batch_limit.saturating_mul(ideal_line_batch_kbytes);

        if config::verbose() {
            eprintln!(
                "[CsvReaderBuilder]: prefetch_limit: {}, \
                line_batch_count_limit: {}, \
                line_batch_kbytes_limit: {}",
                self.prefetch_limit.load(),
                line_batch_count_limit,
                line_batch_kbytes_limit,
            );
        }

        self.prefetch_semaphore
            .set(Arc::new(tokio::sync::Semaphore::new(prefetch_limit)))
            .unwrap();
        self.line_batch_budget
            .set(PipelineBudget::new(
                line_batch_count_limit,
                line_batch_kbytes_limit,
            ))
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
                DynByteSourceBuilder::ObjectStore(FetchConfig::streaming())
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
            line_batch_budget: self.line_batch_budget.get().unwrap().clone(),
            init_data: None,
            io_metrics: OptIOMetrics(self.io_metrics.get().cloned()),
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
