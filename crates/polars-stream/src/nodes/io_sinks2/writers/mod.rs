use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_plan::dsl::FileType;
use polars_utils::IdxSize;

use crate::nodes::io_sinks2::writers::interface::FileWriterStarter;

#[cfg(feature = "csv")]
mod csv;
pub mod interface;
#[cfg(feature = "ipc")]
mod ipc;
#[cfg(feature = "json")]
mod ndjson;
#[cfg(feature = "parquet")]
mod parquet;

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
                row_group_size: options
                    .row_group_size
                    .map(|x| IdxSize::try_from(x).unwrap()),
            }) as _
        },
        #[cfg(feature = "ipc")]
        FileType::Ipc(options) => {
            use crate::nodes::io_sinks2::writers::ipc::IpcWriterStarter;

            Arc::new(IpcWriterStarter {
                options: *options,
                schema: file_schema.clone(),
                pipeline_depth,
                sync_on_close,
                record_batch_size: options
                    .record_batch_size
                    .map(|x| IdxSize::try_from(x).unwrap()),
            }) as _
        },
        #[cfg(feature = "csv")]
        FileType::Csv(options) => {
            use polars_io::prelude::CsvSerializer;

            use crate::nodes::io_sinks2::writers::csv::CsvWriterStarter;

            Arc::new(CsvWriterStarter {
                options: options.clone().into(),
                base_serializer: CsvSerializer::new(
                    file_schema.clone(),
                    Arc::new(options.serialize_options.clone()),
                )?
                .into(),
                schema: file_schema.clone(),
                pipeline_depth,
                sync_on_close,
                initialized_state: Default::default(),
            }) as _
        },
        #[cfg(feature = "json")]
        FileType::Json(polars_io::json::JsonWriterOptions {}) => {
            use crate::nodes::io_sinks2::writers::ndjson::NDJsonWriterStarter;

            Arc::new(NDJsonWriterStarter {
                schema: file_schema.clone(),
                pipeline_depth,
                sync_on_close,
                initialized_state: Default::default(),
            }) as _
        },
        #[cfg(not(any(
            feature = "parquet",
            feature = "ipc",
            feature = "csv",
            feature = "json"
        )))]
        _ => panic!("no enum variants on FileType (hint: missing feature flags?)"),
    })
}
