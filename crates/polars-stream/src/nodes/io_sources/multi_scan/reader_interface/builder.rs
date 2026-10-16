//! Interface for single-file readers

use std::fmt::Debug;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;
use polars_utils::pl_str::PlSmallStr;

use super::FileReader;
use super::capabilities::ReaderCapabilities;
use crate::execute::StreamingExecutionState;
use crate::metrics::IOMetrics;

pub trait FileReaderBuilder: Debug + Send + Sync + 'static {
    fn reader_name(&self) -> PolarsResult<PlSmallStr>;

    fn reader_capabilities(&self) -> PolarsResult<ReaderCapabilities>;

    /// Used by readers that need access to `StreamingExecutionState`.
    fn set_execution_state(&self, _execution_state: &StreamingExecutionState) {}

    fn set_io_metrics(&self, _io_metrics: Arc<IOMetrics>) {}

    #[cfg(feature = "python")]
    fn downcast_as_external_python_reader(
        &self,
    ) -> Option<&polars_io::external_reader::python::PythonFileReaderBuilder> {
        None
    }

    fn is_external_python_reader(&self) -> bool {
        false
    }

    fn is_external_python_reader_with_filter_support(&self) -> PolarsResult<bool> {
        Ok(false)
    }

    /// Preferred number of readers to initialize concurrently for remote sources, if it differs
    /// from the default. Scans over many small remote files are latency-bound, readers with cheap
    /// initialization can prefer a higher number. The multi-scan applies overrides and limits.
    fn preferred_remote_pre_init(&self) -> Option<usize> {
        None
    }

    /// For readers with a pipeline budget: a guard that stops counting the scan as running when
    /// dropped. Call after `set_execution_state`.
    #[cfg(any(feature = "parquet", feature = "ipc"))]
    fn active_scan(&self) -> Option<crate::nodes::io_sources::shared::pipeline_budget::ActiveScan> {
        None
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        scan_source_idx: usize,
    ) -> PolarsResult<Box<dyn FileReader>>;
}
