use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use polars_async::executor::{self, TaskPriority};
use polars_async::primitives::connector;
use polars_error::PolarsResult;
use polars_io::metrics::IOMetrics;
use polars_plan::dsl::UnifiedSinkArgs;
use polars_utils::index::NonZeroIdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::execute::StreamingExecutionState;
use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::error_capture::ErrorCapture;
use crate::nodes::io_sinks::components::file_provider::FileProvider;
use crate::nodes::io_sinks::components::partition_distributor::PartitionDistributor;
use crate::nodes::io_sinks::components::partition_morsel_sender::PartitionMorselSender;
use crate::nodes::io_sinks::components::partition_sink_starter::PartitionSinkStarter;
use crate::nodes::io_sinks::components::partitioner::Partitioner;
use crate::nodes::io_sinks::components::partitioner_pipeline::PartitionerPipeline;
use crate::nodes::io_sinks::components::sinked_path_info_list::{
    SinkedPathInfoList, call_sinked_paths_callback,
};
use crate::nodes::io_sinks::components::size::{NonZeroRowCountAndSize, SplitMode};
use crate::nodes::io_sinks::config::{IOSinkNodeConfig, IOSinkTarget, PartitionedTarget};
use crate::nodes::io_sinks::writers::create_file_writer_starter;
use crate::nodes::io_sinks::writers::interface::FileWriterStarter;

pub fn start_partition_sink_pipeline(
    node_name: &PlSmallStr,
    morsel_rx: connector::Receiver<Morsel>,
    config: IOSinkNodeConfig,
    execution_state: &StreamingExecutionState,
    io_metrics: Option<Arc<IOMetrics>>,
) -> PolarsResult<executor::AbortOnDropHandle<PolarsResult<()>>> {
    let num_pipelines: NonZeroUsize = execution_state.num_pipelines.try_into().unwrap();

    let inflight_morsel_limit = config.inflight_morsel_limit(num_pipelines);
    let num_pipelines_per_sink = config.num_pipelines_per_sink(num_pipelines);
    let max_open_sinks = config.max_open_sinks().get();
    let upload_chunk_size = config.partitioned_upload_chunk_size();
    let upload_max_concurrency = config.partitioned_upload_concurrency();
    let bytes_bufferer_config = config.bytes_bufferer_config();

    let IOSinkNodeConfig {
        file_format,
        target: IOSinkTarget::Partitioned(target),
        unified_sink_args:
            UnifiedSinkArgs {
                mkdir: _,
                maintain_order: _,
                sync_on_close,
                cloud_options,
                sinked_paths_callback,
            },
        input_schema: _,
    } = config
    else {
        unreachable!()
    };

    let PartitionedTarget {
        base_path,
        mut file_path_provider,
        partitioner,
        hstack_keys,
        include_keys_in_file,
        file_schema,
        file_size_limit,
    } = *target;

    let node_name = node_name.clone();
    let verbose = polars_core::config::verbose();
    let in_memory_exec_state = Arc::new(execution_state.in_memory_exec_state.clone());
    let io_metrics_is_some = io_metrics.is_some();

    if let Some(file_part_prefix) = file_path_provider.file_part_prefix_mut() {
        use std::fmt::Write as _;
        let uuid = uuid::Uuid::now_v7();
        let uuid = uuid.as_simple();
        write!(file_part_prefix, "{uuid}").unwrap();
    }

    let sinked_path_info_list: Option<SinkedPathInfoList> = sinked_paths_callback
        .is_some()
        .then(SinkedPathInfoList::default);

    let file_provider = Arc::new(FileProvider {
        base_path,
        cloud_options,
        provider_type: file_path_provider,
        upload_chunk_size,
        upload_max_concurrency,
        io_metrics,
        sinked_path_info_list: sinked_path_info_list.clone(),
    });

    let file_writer_starter: Arc<dyn FileWriterStarter> =
        create_file_writer_starter(&file_format, &file_schema, bytes_bufferer_config)?;

    let mut target_sink_morsel_size = file_writer_starter.target_sink_morsel_size();

    if let Some(file_size_limit) = file_size_limit {
        target_sink_morsel_size.target_num_rows = NonZeroIdxSize::min(
            target_sink_morsel_size.target_num_rows,
            file_size_limit.num_rows,
        );
        target_sink_morsel_size.target_num_bytes = NonZeroU64::min(
            target_sink_morsel_size.target_num_bytes,
            file_size_limit.num_bytes,
        );
        target_sink_morsel_size.target_num_rows_mode = SplitMode::Exact;
    }

    if verbose {
        eprintln!(
            "{node_name}: start_partition_sink_pipeline: \
            partitioner: {}, \
            file_writer_starter: {}, \
            file_provider: {:?}, \
            max_open_sinks: {}, \
            inflight_morsel_limit: {}, \
            target_sink_morsel_size: {:?}, \
            file_size_limit: {:?}, \
            upload_chunk_size: {}, \
            upload_concurrency: {}, \
            io_metrics: {}, \
            build_sinked_path_info_list: {}",
            partitioner.verbose_display(),
            file_writer_starter.writer_name(),
            &file_provider.provider_type,
            max_open_sinks,
            inflight_morsel_limit,
            target_sink_morsel_size,
            file_size_limit,
            upload_chunk_size,
            upload_max_concurrency,
            io_metrics_is_some,
            sinked_path_info_list.is_some(),
        );
    }

    let (partitioned_dfs_tx, partitioned_dfs_rx) = tokio::sync::mpsc::channel(match &partitioner {
        Partitioner::Keyed(_) => inflight_morsel_limit.get(),
        Partitioner::FileSize => 1,
    });
    let inflight_morsel_semaphore =
        Arc::new(tokio::sync::Semaphore::new(inflight_morsel_limit.get()));
    let no_partition_keys = matches!(partitioner, Partitioner::FileSize);

    let partitioner_handle = executor::AbortOnDropHandle::new(executor::spawn(
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
        sync_on_close,
        num_pipelines_per_sink,
    };

    let partition_morsel_sender = PartitionMorselSender {
        target_sink_morsel_size,
        file_size_limit: file_size_limit.unwrap_or(NonZeroRowCountAndSize::MAX),
        inflight_morsel_semaphore,
        open_sinks_semaphore: open_sinks_semaphore.clone(),
        partition_sink_starter: partition_sink_starter.clone(),
        hstack_keys: hstack_keys.filter(|_| include_keys_in_file),
        error_capture: error_capture.clone(),
    };

    let partition_distributor_handle = executor::AbortOnDropHandle::new(executor::spawn(
        TaskPriority::High,
        PartitionDistributor {
            node_name: node_name.clone(),
            partitioned_dfs_rx,
            partition_morsel_sender,
            error_capture,
            error_handle,
            max_open_sinks,
            open_sinks_semaphore,
            partition_sink_starter,
            no_partition_keys,
            verbose,
        }
        .run(),
    ));

    let handle = executor::AbortOnDropHandle::new(executor::spawn(TaskPriority::Low, async move {
        partitioner_handle.await;
        partition_distributor_handle.await?;

        if let Some(sinked_paths_callback) = sinked_paths_callback {
            if verbose {
                eprintln!("{node_name}: Call sinked path info callback");
            }

            call_sinked_paths_callback(sinked_paths_callback, sinked_path_info_list.unwrap())
                .await?;
        }

        Ok(())
    }));

    Ok(handle)
}
