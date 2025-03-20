//! Interface for single-file readers

use std::fmt::Debug;
use std::sync::Arc;

use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::FileReader;

/// `FileReaderType` to avoid confusion with a `FileType` enum from polars-plan.
#[derive(Debug, Clone, PartialEq)]
pub enum FileReaderType {
    #[cfg(feature = "parquet")]
    Parquet,
    #[cfg(feature = "ipc")]
    #[expect(unused)]
    Ipc,
    #[cfg(feature = "csv")]
    #[expect(unused)]
    Csv,
    #[cfg(feature = "json")]
    NDJson,
    /// So that we can compile when all feature flags disabled.
    #[expect(unused)]
    Unknown,
}

pub trait FileReaderBuilder: Debug + Send + Sync + 'static {
    fn file_type(&self) -> FileReaderType;

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
    ) -> Box<dyn FileReader>;
}
