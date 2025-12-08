use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::FileType;

use crate::nodes::io_sinks2::writers::interface::FileWriterStarter;

pub mod interface;
#[cfg(feature = "parquet")]
pub mod parquet;

pub fn create_file_writer_starter(
    file_format: &Arc<FileType>,
    file_schema: &SchemaRef,
    pipeline_depth: usize,
    sync_on_close: SyncOnCloseType,
) -> PolarsResult<Arc<dyn FileWriterStarter>> {
    Ok(match file_format.as_ref() {
        #[cfg(feature = "parquet")]
        FileType::Parquet(options) => {
            use polars_core::prelude::CompatLevel;
            use polars_io::schema_to_arrow_checked;

            use crate::nodes::io_sinks2::writers::parquet::ParquetWriterStarter;

            let arrow_schema = Arc::new(schema_to_arrow_checked(
                file_schema.as_ref(),
                CompatLevel::newest(),
                "",
            )?);

            Arc::new(ParquetWriterStarter {
                options: options.clone(),
                arrow_schema,
                initialized_state: Default::default(),
                pipeline_depth,
                sync_on_close,
            }) as _
        },
        _ => unimplemented!(),
    })
}
