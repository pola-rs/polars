use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::io::ipc::read::FileMetadata;
use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_plan::dsl::ScanSource;
use polars_utils::relaxed_cell::RelaxedCell;

use super::{DynByteSourceBuilder, IpcFileReader};
use crate::async_primitives::wait_group::WaitGroup;
#[cfg(feature = "ipc")]
use crate::metrics::IOMetrics;
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

pub struct IpcReaderBuilder {
    pub first_metadata: Option<Arc<FileMetadata>>,
    pub options: Arc<IpcScanOptions>,
    pub prefetch_limit: RelaxedCell<usize>,
    pub prefetch_semaphore: std::sync::OnceLock<Arc<tokio::sync::Semaphore>>,
    pub shared_prefetch_wait_group_slot: Arc<std::sync::Mutex<Option<WaitGroup>>>,
    pub io_metrics: std::sync::OnceLock<Arc<IOMetrics>>,
}

impl std::fmt::Debug for IpcReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcBuilder")
            .field("first_metadata", &self.first_metadata)
            .field("options", &self.options)
            .field("prefetch_semaphore", &self.prefetch_semaphore)
            .finish()
    }
}

#[cfg(feature = "ipc")]
impl FileReaderBuilder for IpcReaderBuilder {
    fn reader_name(&self) -> &str {
        "ipc"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        RC::ROW_INDEX | RC::PRE_SLICE
    }

    fn set_execution_state(&self, execution_state: &crate::execute::StreamingExecutionState) {
        let prefetch_limit = std::env::var("POLARS_RECORD_BATCH_PREFETCH_SIZE")
            .map(|x| {
                x.parse::<NonZeroUsize>()
                    .unwrap_or_else(|_| {
                        panic!("invalid value for POLARS_RECORD_BATCH_PREFETCH_SIZE: {x}")
                    })
                    .get()
            })
            .unwrap_or(execution_state.num_pipelines.saturating_mul(2))
            .max(1);

        self.prefetch_limit.store(prefetch_limit);

        if config::verbose() {
            eprintln!(
                "[IpcReaderBuilder]: prefetch_limit: {}",
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
        scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        use crate::metrics::OptIOMetrics;
        use crate::nodes::io_sources::ipc::RecordBatchPrefetchSync;

        let scan_source = source;
        let config = self.options.clone();
        let verbose = config::verbose();

        let metadata = if scan_source_idx == 0 {
            self.first_metadata.clone()
        } else {
            None
        };

        let byte_source_builder =
            if scan_source.is_cloud_url() || polars_config::config().force_async() {
                DynByteSourceBuilder::ObjectStore
            } else {
                DynByteSourceBuilder::Mmap
            };

        let reader = IpcFileReader {
            scan_source,
            cloud_options,
            config,
            metadata,
            byte_source_builder,
            record_batch_prefetch_sync: RecordBatchPrefetchSync {
                prefetch_limit: self.prefetch_limit.load(),
                prefetch_semaphore: Arc::clone(self.prefetch_semaphore.get().unwrap()),
                shared_prefetch_wait_group_slot: Arc::clone(&self.shared_prefetch_wait_group_slot),
                prev_all_spawned: None,
                current_all_spawned: None,
            },
            io_metrics: OptIOMetrics(self.io_metrics.get().cloned()),
            verbose,
            init_data: None,
            checked: self.options.checked,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
