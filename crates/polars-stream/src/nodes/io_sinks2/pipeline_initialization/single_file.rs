use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::pl_async;
use polars_plan::dsl::UnifiedSinkArgs;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::morsel::Morsel;
use crate::nodes::io_sinks2::components::morsel_resize_pipeline::MorselResizePipeline;
use crate::nodes::io_sinks2::config::{IOSinkNodeConfig, IOSinkTarget};
use crate::nodes::io_sinks2::writers::create_file_writer_starter;
use crate::nodes::io_sinks2::writers::interface::FileWriterStarter;
use crate::utils::tokio_handle_ext;

pub fn start_single_file_sink_pipeline(
    node_name: PlSmallStr,
    morsel_rx: connector::Receiver<Morsel>,
    config: IOSinkNodeConfig,
) -> PolarsResult<async_executor::AbortOnDropHandle<PolarsResult<()>>> {
    let inflight_morsel_limit = config.inflight_morsel_limit();
    let per_sink_pipeline_depth = config.per_sink_pipeline_depth();
    let upload_chunk_size = config.cloud_upload_chunk_size();

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
        num_pipelines: _,
    } = config
    else {
        unreachable!()
    };

    let file_schema = input_schema;
    let verbose = polars_core::config::verbose();

    let file_open_task =
        tokio_handle_ext::AbortOnDropHandle(pl_async::get_runtime().spawn(async move {
            target
                .open_into_writeable_async(cloud_options.as_deref(), mkdir, upload_chunk_size)
                .await
        }));

    let file_writer_starter: Arc<dyn FileWriterStarter> = create_file_writer_starter(
        &file_format,
        &file_schema,
        per_sink_pipeline_depth,
        sync_on_close,
    )?;
    let ideal_morsel_size = file_writer_starter.ideal_morsel_size();

    if verbose {
        eprintln!(
            "{node_name}: start_single_file_sink_pipeline: \
            file_writer_starter: {}, \
            ideal_morsel_size: {:?}, \
            upload_chunk_size: {}",
            file_writer_starter.writer_name(),
            ideal_morsel_size,
            upload_chunk_size
        )
    }

    let (writer_tx, writer_rx) = connector::connector();
    let writer_handle = file_writer_starter.start_file_writer(writer_rx, file_open_task)?;

    let empty_with_schema_df = DataFrame::empty_with_arc_schema(file_schema.clone());
    let inflight_morsel_semaphore = Arc::new(tokio::sync::Semaphore::new(inflight_morsel_limit));

    let resize_pipeline = MorselResizePipeline {
        empty_with_schema_df,
        ideal_morsel_size,
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
