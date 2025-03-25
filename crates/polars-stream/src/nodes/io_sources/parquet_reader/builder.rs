use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_plan::dsl::ScanSource;

use super::{FileReader, ParquetFileReader};
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;

#[cfg(feature = "parquet")]
impl FileReaderBuilder for Arc<polars_io::parquet::read::ParquetOptions> {
    fn reader_name(&self) -> &str {
        "parquet"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE | RC::FILTER
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let config = self.clone();
        let verbose = config::verbose();

        let byte_source_builder = if scan_source.is_cloud_url() || config::force_async() {
            DynByteSourceBuilder::ObjectStore
        } else {
            DynByteSourceBuilder::Mmap
        };

        let reader = ParquetFileReader {
            scan_source,
            cloud_options,
            config,
            byte_source_builder,
            verbose,

            init_data: None,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
