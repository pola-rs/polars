use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::FileType;

use crate::nodes::io_sinks2::writers::interface::FileWriterStarter;

pub mod interface;

#[expect(unused)]
pub fn create_file_writer_starter(
    file_format: &Arc<FileType>,
    file_schema: &SchemaRef,
    pipeline_depth: usize,
    sync_on_close: SyncOnCloseType,
) -> PolarsResult<Arc<dyn FileWriterStarter>> {
    unimplemented!()
}
