use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::{FileReader, NDJsonFileReader};
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;

#[cfg(feature = "json")]
impl FileReaderBuilder for Arc<polars_plan::dsl::NDJsonReadOptions> {
    fn reader_name(&self) -> &str {
        "ndjson"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let options = self.clone();
        let verbose = config::verbose();

        let reader = NDJsonFileReader {
            scan_source,
            cloud_options,
            options,
            cached_bytes: None,
            verbose,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
