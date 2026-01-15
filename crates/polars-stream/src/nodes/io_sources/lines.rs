use std::sync::Arc;

use polars_core::config;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;

use crate::nodes::io_sources::multi_scan::reader_interface::FileReader;
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::ndjson::NDJsonFileReader;
use crate::nodes::io_sources::ndjson::builder::ndjson_reader_capabilities;
use crate::nodes::io_sources::ndjson::chunk_reader::ChunkReaderBuilder;

#[derive(Debug)]
pub struct LineReaderBuilder {}

impl FileReaderBuilder for LineReaderBuilder {
    fn reader_name(&self) -> &str {
        "line"
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
        let chunk_reader_builder = ChunkReaderBuilder::Lines;
        let verbose = config::verbose();

        let reader = NDJsonFileReader {
            scan_source,
            cloud_options,
            chunk_reader_builder,
            count_rows_fn: polars_io::scan_lines::count_lines,
            cached_bytes: None,
            verbose,
        };

        Box::new(reader) as _
    }
}
