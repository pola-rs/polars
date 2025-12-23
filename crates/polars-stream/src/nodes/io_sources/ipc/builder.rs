use std::sync::Arc;

use arrow::io::ipc::read::FileMetadata;
use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::{DynByteSourceBuilder, IpcFileReader};
use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;

#[derive(Debug)]
pub struct IpcReaderBuilder {
    pub first_metadata: Option<Arc<FileMetadata>>,
}

#[cfg(feature = "ipc")]
impl FileReaderBuilder for IpcReaderBuilder {
    fn reader_name(&self) -> &str {
        "ipc"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        RC::NEEDS_FILE_CACHE_INIT | RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let verbose = config::verbose();

        let metadata = if scan_source_idx == 0 {
            self.first_metadata.clone()
        } else {
            None
        };

        let byte_source_builder = if scan_source.is_cloud_url() || config::force_async() {
            DynByteSourceBuilder::ObjectStore
        } else {
            DynByteSourceBuilder::Mmap
        };

        let reader = IpcFileReader {
            scan_source,
            cloud_options,
            metadata,
            byte_source_builder,
            verbose,
            init_data: None,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}
