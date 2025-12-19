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
    #[expect(unused)]
    pub first_metadata: Option<Arc<FileMetadata>>,
}

#[cfg(feature = "ipc")]
impl FileReaderBuilder for IpcReaderBuilder {
    fn reader_name(&self) -> &str {
        "ipc"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;

        RC::NEEDS_FILE_CACHE_INIT | RC::ROW_INDEX | RC::PRE_SLICE //kdn TODO | RC::NEGATIVE_PRE_SLICE
        //kdn TODO: NEG if m-mapped
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        #[expect(unused)] scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let verbose = config::verbose();

        // kdn TODO: review this FIXME
        //
        // FIXME: For some reason the metadata does not match on idx == 0, and we end up with
        // * ComputeError: out-of-spec: InvalidBuffersLength { buffers_size: 1508, file_size: 763 }
        //
        // let metadata: Option<Arc<FileMetadata>> = if scan_source_idx == 0 {
        //     self.first_metadata.clone()
        // } else {
        //     None
        // };

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
