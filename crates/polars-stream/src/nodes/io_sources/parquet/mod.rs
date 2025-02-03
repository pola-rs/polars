use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use mem_prefetch_funcs::get_memory_prefetch_func;
use polars_core::config;
use polars_core::prelude::ArrowSchema;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::{FileMetadata, ParquetOptions};
use polars_io::utils::byte_source::DynByteSourceBuilder;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{FileInfo, ScanSources};
use polars_plan::prelude::FileScanOptions;
use polars_utils::index::AtomicIdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::TaskPriority;
use crate::utils::task_handles_ext;

mod init;
mod mem_prefetch_funcs;
mod metadata_fetch;
mod metadata_utils;
mod row_group_data_fetch;
mod row_group_decode;

type AsyncTaskData = (
    Vec<crate::async_primitives::distributor_channel::Receiver<(DataFrame, MorselSeq)>>,
    task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
);

#[allow(clippy::type_complexity)]
pub struct ParquetSourceNode {
    scan_sources: ScanSources,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    predicate: Option<ScanIOPredicate>,
    options: ParquetOptions,
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    first_metadata: Option<Arc<FileMetadata>>,
    // Run-time vars
    config: Config,
    verbose: bool,
    schema: Option<Arc<ArrowSchema>>,
    projected_arrow_schema: Option<Arc<ArrowSchema>>,
    byte_source_builder: DynByteSourceBuilder,
    memory_prefetch_func: fn(&[u8]) -> (),
    /// The offset is an AtomicIdxSize, as in the negative slice case, the row
    /// offset becomes relative to the starting point in the list of files,
    /// so the row index offset needs to be updated by the initializer to
    /// reflect this (https://github.com/pola-rs/polars/issues/19607).
    row_index: Option<Arc<(PlSmallStr, AtomicIdxSize)>>,
    // This permit blocks execution until the first morsel is requested.
    morsel_stream_starter: Option<tokio::sync::oneshot::Sender<()>>,
    // This is behind a Mutex so that we can call `shutdown()` asynchronously.
    async_task_data: Arc<tokio::sync::Mutex<Option<AsyncTaskData>>>,
    is_finished: Arc<AtomicBool>,
}

#[derive(Debug)]
struct Config {
    num_pipelines: usize,
    /// Number of files to pre-fetch metadata for concurrently
    metadata_prefetch_size: usize,
    /// Number of files to decode metadata for in parallel in advance
    metadata_decode_ahead_size: usize,
    /// Number of row groups to pre-fetch concurrently, this can be across files
    row_group_prefetch_size: usize,
    /// Minimum number of values for a parallel spawned task to process to amortize
    /// parallelism overhead.
    min_values_per_thread: usize,
}

#[allow(clippy::too_many_arguments)]
impl ParquetSourceNode {
    pub fn new(
        scan_sources: ScanSources,
        file_info: FileInfo,
        predicate: Option<ScanIOPredicate>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        mut file_options: FileScanOptions,
        first_metadata: Option<Arc<FileMetadata>>,
    ) -> Self {
        let verbose = config::verbose();

        let byte_source_builder = if scan_sources.is_cloud_url() || config::force_async() {
            DynByteSourceBuilder::ObjectStore
        } else {
            DynByteSourceBuilder::Mmap
        };
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        let row_index = file_options
            .row_index
            .take()
            .map(|ri| Arc::new((ri.name, AtomicIdxSize::new(ri.offset))));

        Self {
            scan_sources,
            file_info,
            hive_parts: None,
            predicate,
            options,
            cloud_options,
            file_options,
            first_metadata,

            config: Config {
                // Initialized later
                num_pipelines: 0,
                metadata_prefetch_size: 0,
                metadata_decode_ahead_size: 0,
                row_group_prefetch_size: 0,
                min_values_per_thread: 0,
            },
            verbose,
            schema: None,
            projected_arrow_schema: None,
            byte_source_builder,
            memory_prefetch_func,
            row_index,

            morsel_stream_starter: None,
            async_task_data: Arc::new(tokio::sync::Mutex::new(None)),
            is_finished: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ComputeNode for ParquetSourceNode {
    fn name(&self) -> &str {
        "parquet_source"
    }

    fn initialize(&mut self, num_pipelines: usize) {
        self.config = {
            let metadata_prefetch_size = polars_core::config::get_file_prefetch_size();
            // Limit metadata decode to the number of threads.
            let metadata_decode_ahead_size =
                (metadata_prefetch_size / 2).min(1 + num_pipelines).max(1);
            let row_group_prefetch_size = polars_core::config::get_rg_prefetch_size();

            // This can be set to 1 to force column-per-thread parallelism, e.g. for bug reproduction.
            let min_values_per_thread = std::env::var("POLARS_MIN_VALUES_PER_THREAD")
                .map(|x| x.parse::<usize>().expect("integer").max(1))
                .unwrap_or(16_777_216);

            Config {
                num_pipelines,
                metadata_prefetch_size,
                metadata_decode_ahead_size,
                row_group_prefetch_size,
                min_values_per_thread,
            }
        };

        if self.verbose {
            eprintln!("[ParquetSource]: {:?}", &self.config);
        }

        self.schema = Some(self.file_info.reader_schema.take().unwrap().unwrap_left());

        self.init_projected_arrow_schema();

        let (raw_morsel_receivers, raw_morsel_distributor_task_handle) =
            self.init_raw_morsel_distributor();

        self.async_task_data
            .try_lock()
            .unwrap()
            .replace((raw_morsel_receivers, raw_morsel_distributor_task_handle));
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        use std::sync::atomic::Ordering;

        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        if self.is_finished.load(Ordering::Relaxed) {
            send[0] = PortState::Done;
            assert!(
                self.async_task_data.try_lock().unwrap().is_none(),
                "should have already been shut down"
            );
        } else if send[0] == PortState::Done {
            {
                // Early shutdown - our port state was set to `Done` by the downstream nodes.
                self.shutdown_in_background();
            };
            self.is_finished.store(true, Ordering::Relaxed);
        } else {
            send[0] = PortState::Ready
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        use std::sync::atomic::Ordering;

        assert!(recv_ports.is_empty());
        assert_eq!(send_ports.len(), 1);
        assert!(!self.is_finished.load(Ordering::Relaxed));

        let morsel_senders = send_ports[0].take().unwrap().parallel();

        let mut async_task_data_guard = self.async_task_data.try_lock().unwrap();
        let (raw_morsel_receivers, _) = async_task_data_guard.as_mut().unwrap();

        assert_eq!(raw_morsel_receivers.len(), morsel_senders.len());

        if let Some(v) = self.morsel_stream_starter.take() {
            v.send(()).unwrap();
        }
        let is_finished = self.is_finished.clone();

        let source_token = SourceToken::new();
        let task_handles = raw_morsel_receivers
            .drain(..)
            .zip(morsel_senders)
            .map(|(mut raw_morsel_rx, mut morsel_tx)| {
                let is_finished = is_finished.clone();
                let source_token = source_token.clone();
                scope.spawn_task(TaskPriority::Low, async move {
                    let wait_group = WaitGroup::default();
                    loop {
                        let Ok((df, seq)) = raw_morsel_rx.recv().await else {
                            is_finished.store(true, Ordering::Relaxed);
                            break;
                        };

                        let mut morsel = Morsel::new(df, seq, source_token.clone());
                        morsel.set_consume_token(wait_group.token());
                        if morsel_tx.send(morsel).await.is_err() {
                            break;
                        }

                        wait_group.wait().await;
                        if source_token.stop_requested() {
                            break;
                        }
                    }

                    raw_morsel_rx
                })
            })
            .collect::<Vec<_>>();

        drop(async_task_data_guard);

        let async_task_data = self.async_task_data.clone();

        join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
            {
                let mut async_task_data_guard = async_task_data.try_lock().unwrap();
                let (raw_morsel_receivers, _) = async_task_data_guard.as_mut().unwrap();

                for handle in task_handles {
                    raw_morsel_receivers.push(handle.await);
                }
            }

            if self.is_finished.load(Ordering::Relaxed) {
                self.shutdown().await?;
            }

            Ok(())
        }))
    }
}
