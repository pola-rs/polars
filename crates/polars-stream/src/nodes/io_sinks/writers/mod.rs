use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_plan::dsl::FileWriteFormat;
use polars_utils::IdxSize;

use crate::nodes::io_sinks::writers::interface::FileWriterStarter;

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
    file_format: &FileWriteFormat,
    file_schema: &SchemaRef,
) -> PolarsResult<Arc<dyn FileWriterStarter>> {
    Ok(match file_format {
        #[cfg(feature = "parquet")]
        FileWriteFormat::Parquet(options) => {
            use polars_core::prelude::CompatLevel;
            use polars_io::schema_to_arrow_checked;

            use crate::nodes::io_sinks::writers::parquet::ParquetWriterStarter;

            let arrow_schema = if let Some(arrow_schema) = options.arrow_schema.clone() {
                arrow_schema
            } else {
                Arc::new(schema_to_arrow_checked(
                    file_schema.as_ref(),
                    CompatLevel::newest(),
                    "",
                )?)
            };

            Arc::new(ParquetWriterStarter {
                options: Arc::clone(options),
                arrow_schema,
                initialized_state: Default::default(),
                row_group_size: options
                    .row_group_size
                    .map(|x| IdxSize::try_from(x).unwrap()),
            }) as _
        },
        #[cfg(feature = "ipc")]
        FileWriteFormat::Ipc(options) => {
            Arc::new(crate::nodes::io_sinks::writers::ipc::IpcWriterStarter {
                options: Arc::new(*options),
                schema: file_schema.clone(),
                record_batch_size: options
                    .record_batch_size
                    .map(|x| IdxSize::try_from(x).unwrap()),
            }) as _
        },
        #[cfg(feature = "csv")]
        FileWriteFormat::Csv(options) => {
            use polars_io::prelude::CsvSerializer;

            use crate::nodes::io_sinks::writers::csv::CsvWriterStarter;

            Arc::new(CsvWriterStarter {
                options: Arc::new(options.clone()),
                base_serializer: CsvSerializer::new(
                    file_schema.clone(),
                    Arc::clone(&options.serialize_options),
                )?
                .into(),
                schema: file_schema.clone(),
                initialized_state: Default::default(),
            }) as _
        },
        #[cfg(feature = "json")]
        FileWriteFormat::NDJson(options) => Arc::new(
            crate::nodes::io_sinks::writers::ndjson::NDJsonWriterStarter {
                options: *options,
                schema: file_schema.clone(),
                initialized_state: Default::default(),
            },
        ) as _,
        #[cfg(not(any(
            feature = "parquet",
            feature = "ipc",
            feature = "csv",
            feature = "json"
        )))]
        _ => panic!("no enum variants on FileType (hint: missing feature flags?)"),
    })
}
