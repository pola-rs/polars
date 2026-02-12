use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::metrics::IOMetrics;
use polars_io::pl_async;
use polars_plan::dsl::UnifiedSinkArgs;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::execute::StreamingExecutionState;
use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::morsel_resize_pipeline::MorselResizePipeline;
use crate::nodes::io_sinks::config::{IOSinkNodeConfig, IOSinkTarget};
use crate::nodes::io_sinks::writers::create_file_writer_starter;
use crate::nodes::io_sinks::writers::interface::{FileOpenTaskHandle, FileWriterStarter};
use crate::utils::tokio_handle_ext;

pub fn start_single_file_sink_pipeline(
    node_name: PlSmallStr,
    morsel_rx: connector::Receiver<Morsel>,
    config: IOSinkNodeConfig,
    execution_state: &StreamingExecutionState,
    io_metrics: Option<Arc<IOMetrics>>,
) -> PolarsResult<async_executor::AbortOnDropHandle<PolarsResult<()>>> {
    let num_pipelines: NonZeroUsize = execution_state.num_pipelines.try_into().unwrap();

    let inflight_morsel_limit = config.inflight_morsel_limit(num_pipelines);
    let num_pipelines_per_sink = config.num_pipelines_per_sink(num_pipelines);
    let upload_chunk_size = config.cloud_upload_chunk_size();
    let upload_max_concurrency = config.upload_concurrency();

    let IOSinkNodeConfig {
        file_format,
        target: IOSinkTarget::File(target),
        unified_sink_args:
            UnifiedSinkArgs {
                mkdir,
                maintain_order: _,
                sync_on_close,
                cloud_options,
            },
        input_schema,
    } = config
    else {
        unreachable!()
    };

    let file_schema = input_schema;
    let verbose = polars_core::config::verbose();

    let file_open_task = {
        let io_metrics = io_metrics.clone();
        tokio_handle_ext::AbortOnDropHandle(pl_async::get_runtime().spawn(async move {
            target
                .open_into_writeable_async(
                    cloud_options.as_deref(),
                    mkdir,
                    upload_chunk_size,
                    upload_max_concurrency.get(),
                    io_metrics,
                )
                .await
        }))
    };
    let file_open_task = FileOpenTaskHandle::new(file_open_task, sync_on_close);

    let file_writer_starter: Arc<dyn FileWriterStarter> =
        create_file_writer_starter(&file_format, &file_schema)?;
    let takeable_rows_provider = file_writer_starter.takeable_rows_provider();

    if verbose {
        eprintln!(
            "{node_name}: start_single_file_sink_pipeline: \
            file_writer_starter: {}, \
            takeable_rows_provider: {:?}, \
            inflight_morsel_limit: {}, \
            upload_chunk_size: {}, \
            upload_concurrency: {}, \
            io_metrics: {}",
            file_writer_starter.writer_name(),
            takeable_rows_provider,
            inflight_morsel_limit,
            upload_chunk_size,
            upload_max_concurrency,
            io_metrics.is_some(),
        )
    }

    let (writer_tx, writer_rx) = connector::connector();
    let writer_handle =
        file_writer_starter.start_file_writer(writer_rx, file_open_task, num_pipelines_per_sink)?;

    let empty_with_schema_df = DataFrame::empty_with_arc_schema(file_schema.clone());
    let inflight_morsel_semaphore =
        Arc::new(tokio::sync::Semaphore::new(inflight_morsel_limit.get()));

    let resize_pipeline = MorselResizePipeline {
        empty_with_schema_df,
        takeable_rows_provider,
        inflight_morsel_semaphore,
        morsel_rx,
        morsel_tx: writer_tx,
    };

    let resize_pipeline_handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::High,
        resize_pipeline.run(),
    ));

    let handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::High,
        async move {
            writer_handle.await?;
            let sent_size = resize_pipeline_handle.await?;

            if verbose {
                eprintln!("{node_name}: Statistics: total_size: {sent_size:?}");
            }

            Ok(())
        },
    ));

    Ok(handle)
}
