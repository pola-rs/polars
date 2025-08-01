//! Interface for single-file readers

use std::fmt::Debug;
use std::sync::Arc;

use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::FileReader;
use super::capabilities::ReaderCapabilities;
use crate::execute::StreamingExecutionState;

pub trait FileReaderBuilder: Debug + Send + Sync + 'static {
    fn reader_name(&self) -> &str;

    fn reader_capabilities(&self) -> ReaderCapabilities;

    /// Used by readers that need access to `StreamingExecutionState`.
    fn set_execution_state(&self, _execution_state: &StreamingExecutionState) {}

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        scan_source_idx: usize,
    ) -> Box<dyn FileReader>;
}
