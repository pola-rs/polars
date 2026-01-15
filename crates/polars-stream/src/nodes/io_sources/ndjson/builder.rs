use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use super::{FileReader, NDJsonFileReader};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::ndjson::chunk_reader::ChunkReaderBuilder;

pub fn ndjson_reader_capabilities() -> ReaderCapabilities {
    use ReaderCapabilities as RC;

    RC::NEEDS_FILE_CACHE_INIT | RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE
}

#[cfg(feature = "json")]
impl FileReaderBuilder for polars_plan::dsl::NDJsonReadOptions {
    fn reader_name(&self) -> &str {
        "ndjson"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        ndjson_reader_capabilities()
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let chunk_reader_builder = ChunkReaderBuilder::NDJson {
            ignore_errors: self.ignore_errors,
        };
        let verbose = config::verbose();

        let reader = NDJsonFileReader {
            scan_source,
            cloud_options,
            chunk_reader_builder,
            count_rows_fn: polars_io::ndjson::count_rows,
            cached_bytes: None,
            verbose,
        };

        Box::new(reader) as _
    }
}
