//! Interface for single-file readers

use std::fmt::Debug;
use std::sync::Arc;

use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::FileReader;

/// `FileReaderType` to avoid confusion with a `FileType` enum from polars-plan.
#[derive(Debug, PartialEq)]
pub enum FileReaderType {
    #[cfg(feature = "parquet")]
    #[allow(unused)] // TODO
    Parquet,
    #[cfg(feature = "ipc")]
    #[allow(unused)] // TODO
    Ipc,
    #[cfg(feature = "csv")]
    #[allow(unused)] // TODO
    Csv,
    #[cfg(feature = "json")]
    #[allow(unused)] // TODO
    NDJson,
}

pub trait FileReaderBuilder: Debug + Send + Sync + 'static {
    fn file_type(&self) -> FileReaderType;

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
    ) -> Box<dyn FileReader>;
}

#[cfg(feature = "json")]
impl FileReaderBuilder for Arc<polars_plan::dsl::NDJsonReadOptions> {
    fn file_type(&self) -> FileReaderType {
        FileReaderType::NDJson
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
    ) -> Box<dyn FileReader> {
        Box::new(crate::nodes::io_sources::ndjson::NDJsonFileReader::new(
            source,
            cloud_options,
            self.clone(),
        )) as Box<dyn FileReader>
    }
}
