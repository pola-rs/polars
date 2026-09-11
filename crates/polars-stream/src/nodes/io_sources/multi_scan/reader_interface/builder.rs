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

    fn is_external_python_reader(&self) -> bool {
        false
    }

    fn is_external_python_reader_with_filter_support(&self) -> PolarsResult<bool> {
        Ok(false)
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        scan_source_idx: usize,
    ) -> PolarsResult<Box<dyn FileReader>>;
}
