use std::path::PathBuf;
use std::sync::Arc;

use polars_core::prelude::PlHashMap;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::{FileType, PartitionVariant, SinkOptions};
use polars_utils::pl_str::PlSmallStr;

use super::SinkNode;

pub mod max_size;

pub type ArgsToPathFn = Arc<
    dyn Send
        + Sync
        + Fn(PlHashMap<PlSmallStr, PlSmallStr>) -> (PathBuf, PlHashMap<PlSmallStr, PlSmallStr>),
>;
pub type CreateNewSinkFn = Arc<
    dyn Send
        + Sync
        + Fn(
            SchemaRef,
            PlHashMap<PlSmallStr, PlSmallStr>,
        ) -> PolarsResult<(
            PathBuf,
            Box<dyn SinkNode + Send + Sync>,
            PlHashMap<PlSmallStr, PlSmallStr>,
        )>,
>;

pub fn get_args_to_path_fn(
    variant: &PartitionVariant,
    path_f_string: Arc<PathBuf>,
) -> ArgsToPathFn {
    match variant {
        PartitionVariant::MaxSize(_) => Arc::new(move |args: PlHashMap<PlSmallStr, PlSmallStr>| {
            let part_str = PlSmallStr::from_static("part");
            let path = path_f_string
                .as_ref()
                .display()
                .to_string()
                .replace("{part}", args.get(&part_str).unwrap());
            let path = std::path::PathBuf::from(path);
            (path, args)
        }) as _,
    }
}
pub fn get_create_new_fn(
    file_type: FileType,
    sink_options: SinkOptions,
    args_to_path: ArgsToPathFn,
    cloud_options: Option<CloudOptions>,
) -> CreateNewSinkFn {
    match file_type {
        #[cfg(feature = "ipc")]
        FileType::Ipc(ipc_writer_options) => Arc::new(move |input_schema, args| {
            let (path, args) = args_to_path(args);
            let sink = Box::new(super::ipc::IpcSinkNode::new(
                input_schema,
                path.clone(),
                sink_options.clone(),
                ipc_writer_options,
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok((path, sink, args))
        }) as _,
        #[cfg(feature = "json")]
        FileType::Json(_ndjson_writer_options) => Arc::new(move |_input_schema, args| {
            let (path, args) = args_to_path(args);
            let sink = Box::new(super::json::NDJsonSinkNode::new(
                path.clone(),
                sink_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok((path, sink, args))
        }) as _,
        #[cfg(feature = "parquet")]
        FileType::Parquet(parquet_writer_options) => Arc::new(move |input_schema, args| {
            let (path, args) = args_to_path(args);
            let sink = Box::new(super::parquet::ParquetSinkNode::new(
                input_schema,
                path.as_path(),
                sink_options.clone(),
                &parquet_writer_options,
                cloud_options.clone(),
            )?) as Box<dyn SinkNode + Send + Sync>;
            Ok((path, sink, args))
        }) as _,
        #[cfg(feature = "csv")]
        FileType::Csv(csv_writer_options) => Arc::new(move |input_schema, args| {
            let (path, args) = args_to_path(args);
            let sink = Box::new(super::csv::CsvSinkNode::new(
                path.clone(),
                input_schema,
                sink_options.clone(),
                csv_writer_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok((path, sink, args))
        }) as _,
        #[cfg(not(any(
            feature = "csv",
            feature = "parquet",
            feature = "json",
            feature = "ipc"
        )))]
        _ => {
            panic!("activate source feature")
        },
    }
}
