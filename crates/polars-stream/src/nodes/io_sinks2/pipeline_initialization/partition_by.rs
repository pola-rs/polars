use std::sync::Arc;

use polars_error::PolarsResult;
use polars_plan::dsl::UnifiedSinkArgs;
use polars_utils::pl_str::PlSmallStr;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::execute::StreamingExecutionState;
use crate::morsel::Morsel;
use crate::nodes::io_sinks2::components::arg_sort::ArgSortBy;
use crate::nodes::io_sinks2::components::error_capture::ErrorCapture;
use crate::nodes::io_sinks2::components::file_provider::FileProvider;
use crate::nodes::io_sinks2::components::partition_distributor::PartitionDistributor;
use crate::nodes::io_sinks2::components::partition_morsel_sender::PartitionMorselSender;
use crate::nodes::io_sinks2::components::partition_sink_starter::PartitionSinkStarter;
use crate::nodes::io_sinks2::components::partitioner::Partitioner;
use crate::nodes::io_sinks2::components::partitioner_pipeline::PartitionerPipeline;
use crate::nodes::io_sinks2::config::{IOSinkNodeConfig, IOSinkTarget, PartitionedTarget};
use crate::nodes::io_sinks2::writers::create_file_writer_starter;
use crate::nodes::io_sinks2::writers::interface::FileWriterStarter;

pub fn start_partition_sink_pipeline(
    node_name: &PlSmallStr,
    morsel_rx: connector::Receiver<Morsel>,
    config: IOSinkNodeConfig,
    execution_state: &StreamingExecutionState,
) -> PolarsResult<async_executor::AbortOnDropHandle<PolarsResult<()>>> {
    let inflight_morsel_limit = config.inflight_morsel_limit();
    let per_sink_pipeline_depth = config.per_sink_pipeline_depth();
    let max_open_sinks = config.max_open_sinks();
    let upload_chunk_size = config.partitioned_cloud_upload_chunk_size();

    let IOSinkNodeConfig {
        file_format,
        target: IOSinkTarget::Partitioned(target),
        unified_sink_args:
            UnifiedSinkArgs {
                mkdir: _,
                maintain_order: _,
                sync_on_close,
                cloud_options,
            },
        input_schema: _,
        num_pipelines: _,
    } = config
    else {
        unreachable!()
    };

    let PartitionedTarget {
        base_path,
        file_path_provider,
        partitioner,
        hstack_keys,
        include_keys_in_file,
        file_schema,
        file_size_limit,
        per_partition_sort,
    } = *target;

    let node_name = node_name.clone();
    let verbose = polars_core::config::verbose();
    let in_memory_exec_state = Arc::new(execution_state.in_memory_exec_state.clone());

    let file_provider = Arc::new(FileProvider {
        base_path,
        cloud_options,
        provider_type: file_path_provider,
        upload_chunk_size,
    });

    let file_writer_starter: Arc<dyn FileWriterStarter> = create_file_writer_starter(
        &file_format,
        &file_schema,
        per_sink_pipeline_depth,
        sync_on_close,
    )?;

    let mut ideal_morsel_size = file_writer_starter.ideal_morsel_size();

    if let Some(file_size_limit) = file_size_limit {
        ideal_morsel_size = ideal_morsel_size.min(file_size_limit)
    }

    if verbose {
        eprintln!(
            "{node_name}: start_partition_sink_pipeline: \
            partitioner: {}, \
            file_writer_starter: {}, \
            file_provider: {:?}, \
            max_open_sinks: {}, \
            inflight_morsel_limit: {}, \
            ideal_morsel_size: {:?}, \
            file_size_limit: {:?}, \
            per_partition_sort: {}, \
            upload_chunk_size: {}",
            partitioner.verbose_display(),
            file_writer_starter.writer_name(),
            &file_provider.provider_type,
            max_open_sinks,
            inflight_morsel_limit,
            ideal_morsel_size,
            file_size_limit,
            per_partition_sort.is_some(),
            upload_chunk_size
        );
    }

    let (partitioned_dfs_tx, partitioned_dfs_rx) = tokio::sync::mpsc::channel(match &partitioner {
        Partitioner::Keyed(_) => inflight_morsel_limit,
        Partitioner::FileSize => 1,
    });
    let inflight_morsel_semaphore = Arc::new(tokio::sync::Semaphore::new(inflight_morsel_limit));
    let no_partition_keys = matches!(partitioner, Partitioner::FileSize);

    let partitioner_handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::High,
        PartitionerPipeline {
            morsel_rx,
            partitioner: Arc::new(partitioner),
            inflight_morsel_semaphore: inflight_morsel_semaphore.clone(),
            partitioned_dfs_tx,
            in_memory_exec_state: Arc::clone(&in_memory_exec_state),
        }
        .run(),
    ));

    let (error_capture, error_handle) = ErrorCapture::new();

    let open_sinks_semaphore = Arc::new(tokio::sync::Semaphore::new(max_open_sinks));

    let partition_sink_starter = PartitionSinkStarter {
        file_provider,
        writer_starter: Arc::clone(&file_writer_starter),
    };

    let partition_morsel_sender = PartitionMorselSender {
        ideal_morsel_size,
        file_size_limit,
        inflight_morsel_semaphore,
        open_sinks_semaphore: open_sinks_semaphore.clone(),
        partition_sink_starter: partition_sink_starter.clone(),
        hstack_keys: hstack_keys.filter(|_| include_keys_in_file),
        error_capture: error_capture.clone(),
    };

    let per_partition_sort = per_partition_sort.map(|(by, sort_options)| ArgSortBy {
        by,
        sort_options,
        in_memory_exec_state,
    });

    let partition_distributor_handle =
        async_executor::AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::High,
            PartitionDistributor {
                node_name,
                partitioned_dfs_rx,
                partition_morsel_sender,
                error_capture,
                error_handle,
                max_open_sinks,
                open_sinks_semaphore,
                partition_sink_starter,
                per_partition_sort,
                inflight_morsel_limit,
                no_partition_keys,
                verbose,
            }
            .run(),
        ));

    let handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
        TaskPriority::Low,
        async move {
            partitioner_handle.await;
            partition_distributor_handle.await?;
            Ok(())
        },
    ));

    Ok(handle)
}
