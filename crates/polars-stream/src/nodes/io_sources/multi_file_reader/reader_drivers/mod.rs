mod generic;
mod negative_slice_single;

use std::sync::Arc;

use arrow::bitmap::Bitmap;
use futures::stream::BoxStream;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::ScanSources;
use polars_plan::plans::hive::HivePartitionsDf;

use super::reader_interface::FileReader;
use super::reader_interface::builder::FileReaderType;
use super::reader_interface::output::FileReaderOutputRecv;
use crate::async_primitives::connector::{self};
use crate::nodes::io_sources::multi_file_reader::extra_ops::ExtraOperations;

/// Contains everything a driver should need for its loop.
pub struct DriverState {
    pub file_type: FileReaderType,
    pub sources: ScanSources,
    pub readers_init_iter: BoxStream<'static, PolarsResult<(usize, Box<dyn FileReader>)>>,
    pub reader_port_tx: connector::Sender<FileReaderOutputRecv>,
    pub skip_files_mask: Option<Bitmap>,
    pub extra_ops: ExtraOperations,
    pub hive_parts: Option<Arc<HivePartitionsDf>>,
    pub final_output_schema: SchemaRef,
    pub projected_file_schema: SchemaRef,
    pub num_pipelines: usize,
}
