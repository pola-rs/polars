use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    ArrowSchema, ChunkFull, DataType, IdxCa, InitHashMaps, PlHashMap, StringChunked,
};
use polars_core::schema::IndexOfSchema;
use polars_core::series::{IntoSeries, IsSorted, Series};
use polars_core::utils::operation_exceeded_idxsize_msg;
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_expr::prelude::PhysicalExpr;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::PhysicalIoExpr;
use polars_io::prelude::{FileMetaData, ParquetOptions};
use polars_io::utils::byte_source::{
    ByteSource, DynByteSource, DynByteSourceBuilder, MemSliceByteSource,
};
use polars_io::utils::slice::SplitSlicePosition;
use polars_io::{is_cloud_url, RowIndex};
use polars_parquet::read::RowGroupMetaData;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::FileInfo;
use polars_plan::prelude::FileScanOptions;
use polars_utils::aliases::PlHashSet;
use polars_utils::mmap::MemSlice;
use polars_utils::slice::GetSaferUnchecked;
use polars_utils::IdxSize;

use super::{MorselSeq, TaskPriority};
use crate::async_executor::{self};
use crate::async_primitives::connector::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::get_ideal_morsel_size;
use crate::utils::notify_channel::{notify_channel, NotifyReceiver};
use crate::utils::task_handles_ext;

type AsyncTaskData = Option<(
    Vec<crate::async_primitives::connector::Receiver<(DataFrame, MorselSeq, WaitToken)>>,
    async_executor::AbortOnDropHandle<PolarsResult<()>>,
)>;

#[allow(clippy::type_complexity)]
pub struct ParquetSourceNode {
    paths: Arc<Vec<PathBuf>>,
    file_info: FileInfo,
    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    // Run-time vars
    config: Config,
    verbose: bool,
    physical_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    projected_arrow_fields: Arc<[polars_core::prelude::ArrowField]>,
    byte_source_builder: DynByteSourceBuilder,
    memory_prefetch_func: fn(&[u8]) -> (),
    // This permit blocks execution until the first morsel is requested.
    morsel_stream_starter: Option<tokio::sync::oneshot::Sender<()>>,
    // This is behind a Mutex so that we can call `shutdown()` asynchronously.
    async_task_data: Arc<tokio::sync::Mutex<AsyncTaskData>>,
    row_group_decoder: Option<Arc<RowGroupDecoder>>,
    is_finished: Arc<AtomicBool>,
}

#[allow(clippy::too_many_arguments)]
impl ParquetSourceNode {
    pub fn new(
        paths: Arc<Vec<PathBuf>>,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
    ) -> Self {
        let verbose = config::verbose();

        let byte_source_builder =
            if is_cloud_url(paths[0].to_str().unwrap()) || config::force_async() {
                DynByteSourceBuilder::ObjectStore
            } else {
                DynByteSourceBuilder::Mmap
            };
        let memory_prefetch_func = get_memory_prefetch_func(verbose);

        Self {
            paths,
            file_info,
            hive_parts,
            predicate,
            options,
            cloud_options,
            file_options,

            config: Config {
                // Initialized later
                num_pipelines: 0,
                metadata_prefetch_size: 0,
                metadata_decode_ahead_size: 0,
                row_group_prefetch_size: 0,
            },
            verbose,
            physical_predicate: None,
            projected_arrow_fields: Arc::new([]),
            byte_source_builder,
            memory_prefetch_func,

            morsel_stream_starter: None,
            async_task_data: Arc::new(tokio::sync::Mutex::new(None)),
            row_group_decoder: None,
            is_finished: Arc::new(AtomicBool::new(false)),
        }
    }
}

mod compute_node_impl {

    use std::sync::Arc;

    use polars_expr::prelude::phys_expr_to_io_expr;

    use super::super::compute_node_prelude::*;
    use super::{Config, ParquetSourceNode};
    use crate::morsel::SourceToken;

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

                Config {
                    num_pipelines,
                    metadata_prefetch_size,
                    metadata_decode_ahead_size,
                    row_group_prefetch_size,
                }
            };

            if self.verbose {
                eprintln!("[ParquetSource]: {:?}", &self.config);
            }

            self.init_projected_arrow_fields();
            self.physical_predicate = self.predicate.clone().map(phys_expr_to_io_expr);

            let (raw_morsel_receivers, morsel_stream_task_handle) = self.init_raw_morsel_stream();

            self.async_task_data
                .try_lock()
                .unwrap()
                .replace((raw_morsel_receivers, morsel_stream_task_handle));

            let row_group_decoder = self.init_row_group_decoder();
            self.row_group_decoder = Some(Arc::new(row_group_decoder));
        }

        fn update_state(
            &mut self,
            recv: &mut [PortState],
            send: &mut [PortState],
        ) -> PolarsResult<()> {
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
            recv: &mut [Option<RecvPort<'_>>],
            send: &mut [Option<SendPort<'_>>],
            _state: &'s ExecutionState,
            join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        ) {
            use std::sync::atomic::Ordering;

            assert!(recv.is_empty());
            assert_eq!(send.len(), 1);
            assert!(!self.is_finished.load(Ordering::Relaxed));

            let morsel_senders = send[0].take().unwrap().parallel();

            let mut async_task_data_guard = self.async_task_data.try_lock().unwrap();
            let (raw_morsel_receivers, _) = async_task_data_guard.as_mut().unwrap();

            assert_eq!(raw_morsel_receivers.len(), morsel_senders.len());

            if let Some(v) = self.morsel_stream_starter.take() {
                v.send(()).unwrap();
            }
            let is_finished = self.is_finished.clone();

            let task_handles = raw_morsel_receivers
                .drain(..)
                .zip(morsel_senders)
                .map(|(mut raw_morsel_rx, mut morsel_tx)| {
                    let is_finished = is_finished.clone();

                    scope.spawn_task(TaskPriority::Low, async move {
                        let source_token = SourceToken::new();
                        loop {
                            let Ok((df, morsel_seq, wait_token)) = raw_morsel_rx.recv().await
                            else {
                                is_finished.store(true, Ordering::Relaxed);
                                break;
                            };

                            let mut morsel = Morsel::new(df, morsel_seq, source_token.clone());
                            morsel.set_consume_token(wait_token);

                            if morsel_tx.send(morsel).await.is_err() {
                                break;
                            }

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
}

impl ParquetSourceNode {
    /// # Panics
    /// Panics if called more than once.
    async fn shutdown_impl(
        async_task_data: Arc<tokio::sync::Mutex<AsyncTaskData>>,
        verbose: bool,
    ) -> PolarsResult<()> {
        if verbose {
            eprintln!("[ParquetSource]: Shutting down");
        }

        let (mut raw_morsel_receivers, morsel_stream_task_handle) =
            async_task_data.try_lock().unwrap().take().unwrap();

        raw_morsel_receivers.clear();
        // Join on the producer handle to catch errors/panics.
        // Safety
        // * We dropped the receivers on the line above
        // * This function is only called once.
        morsel_stream_task_handle.await
    }

    fn shutdown(&self) -> impl Future<Output = PolarsResult<()>> {
        if self.verbose {
            eprintln!("[ParquetSource]: Shutdown via `shutdown()`");
        }
        Self::shutdown_impl(self.async_task_data.clone(), self.verbose)
    }

    /// Spawns a task to shut down the source node to avoid blocking the current thread. This is
    /// usually called when data is no longer needed from the source node, as such it does not
    /// propagate any (non-critical) errors. If on the other hand the source node does not provide
    /// more data when requested, then it is more suitable to call [`Self::shutdown`], as it returns
    /// a result that can be used to distinguish between whether the data stream stopped due to an
    /// error or EOF.
    fn shutdown_in_background(&self) {
        if self.verbose {
            eprintln!("[ParquetSource]: Shutdown via `shutdown_in_background()`");
        }
        let async_task_data = self.async_task_data.clone();
        polars_io::pl_async::get_runtime()
            .spawn(Self::shutdown_impl(async_task_data, self.verbose));
    }

    /// Constructs the task that provides a morsel stream.
    #[allow(clippy::type_complexity)]
    fn init_raw_morsel_stream(
        &mut self,
    ) -> (
        Vec<crate::async_primitives::connector::Receiver<(DataFrame, MorselSeq, WaitToken)>>,
        async_executor::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;

        let use_statistics = self.options.use_statistics;

        let (mut raw_morsel_senders, raw_morsel_receivers): (Vec<_>, Vec<_>) =
            (0..self.config.num_pipelines).map(|_| connector()).unzip();

        if let Some((_, 0)) = self.file_options.slice {
            return (
                raw_morsel_receivers,
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    std::future::ready(Ok(())),
                )),
            );
        }

        let reader_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .clone();

        let (normalized_slice_oneshot_rx, metadata_rx, metadata_task_handle) =
            self.init_metadata_fetcher();

        let num_pipelines = self.config.num_pipelines;
        let row_group_prefetch_size = self.config.row_group_prefetch_size;
        let projection = self.file_options.with_columns.clone();
        assert_eq!(self.physical_predicate.is_some(), self.predicate.is_some());
        let predicate = self.physical_predicate.clone();
        let memory_prefetch_func = self.memory_prefetch_func;
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        self.morsel_stream_starter = Some(start_tx);

        let mut row_group_data_fetcher = RowGroupDataFetcher {
            metadata_rx,
            use_statistics,
            verbose,
            reader_schema,
            projection,
            predicate,
            slice_range: None, // Initialized later
            memory_prefetch_func,
            current_path_index: 0,
            current_byte_source: Default::default(),
            current_row_groups: Default::default(),
            current_row_group_idx: 0,
            current_max_row_group_height: 0,
            current_row_offset: 0,
            current_shared_file_state: Default::default(),
        };

        let row_group_decoder = self.init_row_group_decoder();
        let row_group_decoder = Arc::new(row_group_decoder);

        // Processes row group metadata and spawns I/O tasks to fetch row group data. This is
        // currently spawned onto the CPU runtime as it does not directly make any async I/O calls,
        // but instead it potentially performs predicate/slice evaluation on metadata. If we observe
        // that under heavy CPU load scenarios the I/O throughput drops due to this task not being
        // scheduled we can change it to be a high priority task.
        let morsel_stream_task_handle = async_executor::spawn(TaskPriority::Low, async move {
            if start_rx.await.is_err() {
                drop(row_group_data_fetcher);
                return metadata_task_handle.await.unwrap();
            }

            if verbose {
                eprintln!("[ParquetSource]: Starting row group data fetch")
            }

            // We must `recv()` from the `NotifyReceiver` before awaiting on the
            // `normalized_slice_oneshot_rx`, as in the negative offset case the slice resolution
            // only runs after the first notify.
            if !row_group_data_fetcher.init_next_file_state().await {
                drop(row_group_data_fetcher);
                return metadata_task_handle.await.unwrap();
            };

            let slice_range = {
                let Ok(slice) = normalized_slice_oneshot_rx.await else {
                    // If we are here then the producer probably errored.
                    drop(row_group_data_fetcher);
                    return metadata_task_handle.await.unwrap();
                };

                slice.map(|(offset, len)| offset..offset + len)
            };

            row_group_data_fetcher.slice_range = slice_range;

            // Pins a wait group to a channel index.
            struct IndexedWaitGroup {
                index: usize,
                wait_group: WaitGroup,
            }

            impl IndexedWaitGroup {
                async fn wait(self) -> Self {
                    self.wait_group.wait().await;
                    self
                }
            }

            // Ensure proper backpressure by only polling the buffered iterator when a wait group
            // is free.
            let mut wait_groups = (0..num_pipelines)
                .map(|index| {
                    let wait_group = WaitGroup::default();
                    {
                        let _prime_this_wait_group = wait_group.token();
                    }
                    IndexedWaitGroup {
                        index,
                        wait_group: WaitGroup::default(),
                    }
                    .wait()
                })
                .collect::<FuturesUnordered<_>>();

            let mut df_stream = row_group_data_fetcher
                .into_stream()
                .map(|x| async {
                    match x {
                        Ok(handle) => handle.await,
                        Err(e) => Err(e),
                    }
                })
                .buffered(row_group_prefetch_size)
                .map(|x| async {
                    let row_group_decoder = row_group_decoder.clone();

                    match x {
                        Ok(row_group_data) => {
                            async_executor::spawn(TaskPriority::Low, async move {
                                row_group_decoder.row_group_data_to_df(row_group_data).await
                            })
                            .await
                        },
                        Err(e) => Err(e),
                    }
                })
                .buffered(
                    // Because we are using an ordered buffer, we may suffer from head-of-line blocking,
                    // so we add a small amount of buffer.
                    num_pipelines + 4,
                );

            let morsel_seq_ref = &mut MorselSeq::default();
            let mut dfs = vec![].into_iter();

            'main: loop {
                let Some(mut indexed_wait_group) = wait_groups.next().await else {
                    break;
                };

                if dfs.len() == 0 {
                    let Some(v) = df_stream.next().await else {
                        break;
                    };

                    let v = v?;
                    assert!(!v.is_empty());

                    dfs = v.into_iter();
                }

                let mut df = dfs.next().unwrap();
                let morsel_seq = *morsel_seq_ref;
                *morsel_seq_ref = morsel_seq.successor();

                loop {
                    use crate::async_primitives::connector::SendError;

                    let channel_index = indexed_wait_group.index;
                    let wait_token = indexed_wait_group.wait_group.token();

                    match raw_morsel_senders[channel_index].try_send((df, morsel_seq, wait_token)) {
                        Ok(_) => {
                            wait_groups.push(indexed_wait_group.wait());
                            break;
                        },
                        Err(SendError::Closed(v)) => {
                            // The port assigned to this wait group has been closed, so we will not
                            // add it back to the list of wait groups, and we will try to send this
                            // across another port.
                            df = v.0
                        },
                        Err(SendError::Full(_)) => unreachable!(),
                    }

                    let Some(v) = wait_groups.next().await else {
                        // All ports have closed
                        break 'main;
                    };

                    indexed_wait_group = v;
                }
            }

            // Join on the producer handle to catch errors/panics.
            drop(df_stream);
            metadata_task_handle.await.unwrap()
        });

        let morsel_stream_task_handle =
            async_executor::AbortOnDropHandle::new(morsel_stream_task_handle);

        (raw_morsel_receivers, morsel_stream_task_handle)
    }

    /// Constructs the task that fetches file metadata.
    /// Note: This must be called AFTER `self.projected_arrow_fields` has been initialized.
    ///
    /// TODO: During IR conversion the metadata of the first file is already downloaded - see if
    /// we can find a way to re-use it.
    #[allow(clippy::type_complexity)]
    fn init_metadata_fetcher(
        &self,
    ) -> (
        tokio::sync::oneshot::Receiver<Option<(usize, usize)>>,
        NotifyReceiver<(usize, usize, Arc<DynByteSource>, FileMetaData, usize)>,
        task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
    ) {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        assert!(
            !self.projected_arrow_fields.is_empty()
                || self.file_options.with_columns.as_deref() == Some(&[])
        );
        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let needs_max_row_group_height_calc =
            self.file_options.include_file_paths.is_some() || self.hive_parts.is_some();

        let (normalized_slice_oneshot_tx, normalized_slice_oneshot_rx) =
            tokio::sync::oneshot::channel();
        let (metadata_tx, mut metadata_notify_rx, metadata_rx) = notify_channel();

        let byte_source_builder = self.byte_source_builder.clone();

        if self.verbose {
            eprintln!(
                "[ParquetSource]: Byte source builder: {:?}",
                &byte_source_builder
            );
        }

        let fetch_metadata_bytes_for_path_index = {
            let paths = &self.paths;
            let cloud_options = Arc::new(self.cloud_options.clone());

            let paths = paths.clone();
            let cloud_options = cloud_options.clone();
            let byte_source_builder = byte_source_builder.clone();

            move |path_idx: usize| {
                let paths = paths.clone();
                let cloud_options = cloud_options.clone();
                let byte_source_builder = byte_source_builder.clone();

                let handle = io_runtime.spawn(async move {
                    let mut byte_source = Arc::new(
                        byte_source_builder
                            .try_build_from_path(
                                paths[path_idx].to_str().unwrap(),
                                cloud_options.as_ref().as_ref(),
                            )
                            .await?,
                    );
                    let (metadata_bytes, maybe_full_bytes) =
                        read_parquet_metadata_bytes(byte_source.as_ref(), verbose).await?;

                    if let Some(v) = maybe_full_bytes {
                        if !matches!(byte_source.as_ref(), DynByteSource::MemSlice(_)) {
                            if verbose {
                                eprintln!(
                                    "[ParquetSource]: Parquet file was fully fetched during \
                                         metadata read ({} bytes).",
                                    v.len(),
                                );
                            }

                            byte_source = Arc::new(DynByteSource::from(MemSliceByteSource(v)))
                        }
                    }

                    PolarsResult::Ok((path_idx, byte_source, metadata_bytes))
                });

                let handle = task_handles_ext::AbortOnDropHandle(handle);

                std::future::ready(handle)
            }
        };

        let process_metadata_bytes = {
            move |handle: task_handles_ext::AbortOnDropHandle<
                PolarsResult<(usize, Arc<DynByteSource>, MemSlice)>,
            >| {
                let projected_arrow_fields = projected_arrow_fields.clone();
                // Run on CPU runtime - metadata deserialization is expensive, especially
                // for very wide tables.
                let handle = async_executor::spawn(TaskPriority::Low, async move {
                    let (path_index, byte_source, metadata_bytes) = handle.await.unwrap()?;

                    let metadata = polars_parquet::parquet::read::deserialize_metadata(
                        metadata_bytes.as_ref(),
                        metadata_bytes.len() * 2 + 1024,
                    )?;

                    ensure_metadata_has_projected_fields(
                        projected_arrow_fields.as_ref(),
                        &metadata,
                    )?;

                    let file_max_row_group_height = if needs_max_row_group_height_calc {
                        metadata
                            .row_groups
                            .iter()
                            .map(|x| x.num_rows())
                            .max()
                            .unwrap_or(0)
                    } else {
                        0
                    };

                    PolarsResult::Ok((path_index, byte_source, metadata, file_max_row_group_height))
                });

                async_executor::AbortOnDropHandle::new(handle)
            }
        };

        let metadata_prefetch_size = self.config.metadata_prefetch_size;
        let metadata_decode_ahead_size = self.config.metadata_decode_ahead_size;

        let metadata_task_handle = if self
            .file_options
            .slice
            .map(|(offset, _)| offset >= 0)
            .unwrap_or(true)
        {
            normalized_slice_oneshot_tx
                .send(
                    self.file_options
                        .slice
                        .map(|(offset, len)| (offset as usize, len)),
                )
                .unwrap();

            // Safety: `offset + len` does not overflow.
            let slice_range = self
                .file_options
                .slice
                .map(|(offset, len)| offset as usize..offset as usize + len);

            let mut metadata_stream = futures::stream::iter(0..self.paths.len())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            let paths = self.paths.clone();

            // We need to be able to both stop early as well as skip values, which is easier to do
            // using a custom task instead of futures::stream
            io_runtime.spawn(async move {
                let current_row_offset_ref = &mut 0usize;
                let current_path_index_ref = &mut 0usize;

                'main: while metadata_notify_rx.recv().await.is_some() {
                    loop {
                        let current_path_index = *current_path_index_ref;
                        *current_path_index_ref += 1;

                        let Some(v) = metadata_stream.next().await else {
                            break 'main;
                        };

                        let (path_index, byte_source, metadata, file_max_row_group_height) = v
                            .map_err(|err| {
                                err.wrap_msg(|msg| {
                                    format!(
                                        "error at path (index: {}, path: {}): {}",
                                        current_path_index,
                                        paths[current_path_index].to_str().unwrap(),
                                        msg
                                    )
                                })
                            })?;

                        assert_eq!(path_index, current_path_index);

                        let current_row_offset = *current_row_offset_ref;
                        *current_row_offset_ref =
                            current_row_offset.saturating_add(metadata.num_rows);

                        if let Some(slice_range) = slice_range.clone() {
                            match SplitSlicePosition::split_slice_at_file(
                                current_row_offset,
                                metadata.num_rows,
                                slice_range,
                            ) {
                                SplitSlicePosition::Before => {
                                    if verbose {
                                        eprintln!(
                                            "[ParquetSource]: Slice pushdown: \
                                            Skipped file at index {} ({} rows)",
                                            current_path_index, metadata.num_rows
                                        );
                                    }
                                    continue;
                                },
                                SplitSlicePosition::After => unreachable!(),
                                SplitSlicePosition::Overlapping(..) => {},
                            };
                        };

                        {
                            use tokio::sync::mpsc::error::*;
                            match metadata_tx.try_send((
                                path_index,
                                current_row_offset,
                                byte_source,
                                metadata,
                                file_max_row_group_height,
                            )) {
                                Err(TrySendError::Closed(_)) => break 'main,
                                Ok(_) => {},
                                Err(TrySendError::Full(_)) => unreachable!(),
                            }
                        }

                        if let Some(slice_range) = slice_range.as_ref() {
                            if *current_row_offset_ref >= slice_range.end {
                                if verbose {
                                    eprintln!(
                                        "[ParquetSource]: Slice pushdown: \
                                        Stopped reading at file at index {} \
                                        (remaining {} files will not be read)",
                                        current_path_index,
                                        paths.len() - current_path_index - 1,
                                    );
                                }
                                break 'main;
                            }
                        };

                        break;
                    }
                }

                Ok(())
            })
        } else {
            // Walk the files in reverse to translate the slice into a positive offset.
            let slice = self.file_options.slice.unwrap();
            let slice_start_as_n_from_end = -slice.0 as usize;

            let mut metadata_stream = futures::stream::iter((0..self.paths.len()).rev())
                .map(fetch_metadata_bytes_for_path_index)
                .buffered(metadata_prefetch_size)
                .map(process_metadata_bytes)
                .buffered(metadata_decode_ahead_size);

            // Note:
            // * We want to wait until the first morsel is requested before starting this
            let init_negative_slice_and_metadata = async move {
                let mut processed_metadata_rev = vec![];
                let mut cum_rows = 0;

                while let Some(v) = metadata_stream.next().await {
                    let v = v?;
                    let (_, _, metadata, _) = &v;
                    cum_rows += metadata.num_rows;
                    processed_metadata_rev.push(v);

                    if cum_rows >= slice_start_as_n_from_end {
                        break;
                    }
                }

                let (start, len) = if slice_start_as_n_from_end > cum_rows {
                    // We need to trim the slice, e.g. SLICE[offset: -100, len: 75] on a file of 50
                    // rows should only give the first 25 rows.
                    let first_file_position = slice_start_as_n_from_end - cum_rows;
                    (0, slice.1.saturating_sub(first_file_position))
                } else {
                    (cum_rows - slice_start_as_n_from_end, slice.1)
                };

                if len == 0 {
                    processed_metadata_rev.clear();
                }

                normalized_slice_oneshot_tx
                    .send(Some((start, len)))
                    .unwrap();

                let slice_range = start..(start + len);

                PolarsResult::Ok((slice_range, processed_metadata_rev, cum_rows))
            };

            let path_count = self.paths.len();

            io_runtime.spawn(async move {
                // Wait for the first morsel request before we call `init_negative_slice_and_metadata`
                // This also means the receiver must `recv()` once before awaiting on the
                // `normalized_slice_oneshot_rx` to avoid hanging.
                if metadata_notify_rx.recv().await.is_none() {
                    return Ok(());
                }

                let (slice_range, processed_metadata_rev, cum_rows) =
                    async_executor::AbortOnDropHandle::new(async_executor::spawn(
                        TaskPriority::Low,
                        init_negative_slice_and_metadata,
                    ))
                    .await?;

                if verbose {
                    if let Some((path_index, ..)) = processed_metadata_rev.last() {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            begins at file index {}, translated to {:?}",
                            slice, path_index, slice_range
                        );
                    } else {
                        eprintln!(
                            "[ParquetSource]: Slice pushdown: Negatively-offsetted slice {:?} \
                            skipped all files ({} files containing {} rows)",
                            slice, path_count, cum_rows
                        )
                    }
                }

                let mut metadata_iter = processed_metadata_rev.into_iter().rev();
                let current_row_offset_ref = &mut 0usize;

                // do-while: We already consumed a notify above.
                loop {
                    let Some((
                        current_path_index,
                        byte_source,
                        metadata,
                        file_max_row_group_height,
                    )) = metadata_iter.next()
                    else {
                        break;
                    };

                    let current_row_offset = *current_row_offset_ref;
                    *current_row_offset_ref = current_row_offset.saturating_add(metadata.num_rows);

                    assert!(matches!(
                        SplitSlicePosition::split_slice_at_file(
                            current_row_offset,
                            metadata.num_rows,
                            slice_range.clone(),
                        ),
                        SplitSlicePosition::Overlapping(..)
                    ));

                    {
                        use tokio::sync::mpsc::error::*;
                        match metadata_tx.try_send((
                            current_path_index,
                            current_row_offset,
                            byte_source,
                            metadata,
                            file_max_row_group_height,
                        )) {
                            Err(TrySendError::Closed(_)) => break,
                            Ok(v) => v,
                            Err(TrySendError::Full(_)) => unreachable!(),
                        }
                    }

                    if *current_row_offset_ref >= slice_range.end {
                        if verbose {
                            eprintln!(
                                "[ParquetSource]: Slice pushdown: \
                                Stopped reading at file at index {} \
                                (remaining {} files will not be read)",
                                current_path_index,
                                path_count - current_path_index - 1,
                            );
                        }
                        break;
                    }

                    if metadata_notify_rx.recv().await.is_none() {
                        break;
                    }
                }

                Ok(())
            })
        };

        let metadata_task_handle = task_handles_ext::AbortOnDropHandle(metadata_task_handle);

        (
            normalized_slice_oneshot_rx,
            metadata_rx,
            metadata_task_handle,
        )
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_fields`
    /// * `self.physical_predicate`
    fn init_row_group_decoder(&self) -> RowGroupDecoder {
        assert!(
            !self.projected_arrow_fields.is_empty()
                || self.file_options.with_columns.as_deref() == Some(&[])
        );
        assert_eq!(self.predicate.is_some(), self.physical_predicate.is_some());

        let paths = self.paths.clone();
        let hive_partitions = self.hive_parts.clone();
        let hive_partitions_width = hive_partitions
            .as_deref()
            .map(|x| x[0].get_statistics().column_stats().len())
            .unwrap_or(0);
        let include_file_paths = self.file_options.include_file_paths.clone();
        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let row_index = self.file_options.row_index.clone();
        let physical_predicate = self.physical_predicate.clone();
        let ideal_morsel_size = get_ideal_morsel_size();

        RowGroupDecoder {
            paths,
            hive_partitions,
            hive_partitions_width,
            include_file_paths,
            projected_arrow_fields,
            row_index,
            physical_predicate,
            ideal_morsel_size,
        }
    }

    fn init_projected_arrow_fields(&mut self) {
        let reader_schema = self
            .file_info
            .reader_schema
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_left()
            .clone();

        self.projected_arrow_fields =
            if let Some(columns) = self.file_options.with_columns.as_deref() {
                columns
                    .iter()
                    .map(|x| {
                        // `index_of` on ArrowSchema is slow, so we use the polars native Schema,
                        // but we need to remember to subtact the row index.
                        let pos = self.file_info.schema.index_of(x.as_str()).unwrap()
                            - (self.file_options.row_index.is_some() as usize);
                        reader_schema.fields[pos].clone()
                    })
                    .collect()
            } else {
                Arc::from(reader_schema.fields.as_slice())
            };

        if self.verbose {
            eprintln!(
                "[ParquetSource]: {} columns to be projected from {} files",
                self.projected_arrow_fields.len(),
                self.paths.len(),
            );
        }
    }
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
}

/// Represents byte-data that can be transformed into a DataFrame after some computation.
struct RowGroupData {
    byte_source: FetchedBytes,
    path_index: usize,
    row_offset: usize,
    slice: Option<(usize, usize)>,
    file_max_row_group_height: usize,
    row_group_metadata: RowGroupMetaData,
    shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

struct RowGroupDataFetcher {
    metadata_rx: NotifyReceiver<(usize, usize, Arc<DynByteSource>, FileMetaData, usize)>,
    use_statistics: bool,
    verbose: bool,
    reader_schema: Arc<ArrowSchema>,
    projection: Option<Arc<[String]>>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    slice_range: Option<std::ops::Range<usize>>,
    memory_prefetch_func: fn(&[u8]) -> (),
    current_path_index: usize,
    current_byte_source: Arc<DynByteSource>,
    current_row_groups: std::vec::IntoIter<RowGroupMetaData>,
    current_row_group_idx: usize,
    current_max_row_group_height: usize,
    current_row_offset: usize,
    current_shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

fn read_this_row_group(
    rg_md: &RowGroupMetaData,
    predicate: Option<&dyn PhysicalIoExpr>,
    reader_schema: &ArrowSchema,
) -> PolarsResult<bool> {
    let Some(pred) = predicate else {
        return Ok(true);
    };
    use polars_io::prelude::_internal::*;
    // TODO!
    // Optimize this. Now we partition the predicate columns twice. (later on reading as well)
    // I think we must add metadata context where we can cache and amortize the partitioning.
    let mut part_md = PartitionedColumnChunkMD::new(rg_md);
    let live = pred.live_variables();
    part_md.set_partitions(
        live.as_ref()
            .map(|vars| vars.iter().map(|s| s.as_ref()).collect::<PlHashSet<_>>())
            .as_ref(),
    );
    read_this_row_group(Some(pred), &part_md, reader_schema)
}

impl RowGroupDataFetcher {
    fn into_stream(self) -> RowGroupDataStream {
        RowGroupDataStream::new(self)
    }

    async fn init_next_file_state(&mut self) -> bool {
        let Some((path_index, row_offset, byte_source, metadata, file_max_row_group_height)) =
            self.metadata_rx.recv().await
        else {
            return false;
        };

        self.current_path_index = path_index;
        self.current_byte_source = byte_source;
        self.current_max_row_group_height = file_max_row_group_height;
        // The metadata task also sends a row offset to start counting from as it may skip files
        // during slice pushdown.
        self.current_row_offset = row_offset;
        self.current_row_group_idx = 0;
        self.current_row_groups = metadata.row_groups.into_iter();
        self.current_shared_file_state = Default::default();

        true
    }

    async fn next(
        &mut self,
    ) -> Option<PolarsResult<async_executor::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        'main: loop {
            for row_group_metadata in self.current_row_groups.by_ref() {
                let current_row_offset = self.current_row_offset;
                let current_row_group_idx = self.current_row_group_idx;

                let num_rows = row_group_metadata.num_rows();

                self.current_row_offset = current_row_offset.saturating_add(num_rows);
                self.current_row_group_idx += 1;

                if self.use_statistics
                    && !match read_this_row_group(
                        &row_group_metadata,
                        self.predicate.as_deref(),
                        self.reader_schema.as_ref(),
                    ) {
                        Ok(v) => v,
                        Err(e) => return Some(Err(e)),
                    }
                {
                    if self.verbose {
                        eprintln!(
                            "[ParquetSource]: Predicate pushdown: \
                            Skipped row group {} in file {} ({} rows)",
                            current_row_group_idx, self.current_path_index, num_rows
                        );
                    }
                    continue;
                }

                if num_rows > IdxSize::MAX as usize {
                    let msg = operation_exceeded_idxsize_msg(
                        format!("number of rows in row group ({})", num_rows).as_str(),
                    );
                    return Some(Err(polars_err!(ComputeError: msg)));
                }

                let slice = if let Some(slice_range) = self.slice_range.clone() {
                    let (offset, len) = match SplitSlicePosition::split_slice_at_file(
                        current_row_offset,
                        num_rows,
                        slice_range,
                    ) {
                        SplitSlicePosition::Before => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Skipped row group {} in file {} ({} rows)",
                                    current_row_group_idx, self.current_path_index, num_rows
                                );
                            }
                            continue;
                        },
                        SplitSlicePosition::After => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Stop at row group {} in file {} \
                                    (remaining {} row groups will not be read)",
                                    current_row_group_idx,
                                    self.current_path_index,
                                    self.current_row_groups.len(),
                                );
                            };
                            break 'main;
                        },
                        SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    };

                    Some((offset, len))
                } else {
                    None
                };

                let current_byte_source = self.current_byte_source.clone();
                let projection = self.projection.clone();
                let current_shared_file_state = self.current_shared_file_state.clone();
                let memory_prefetch_func = self.memory_prefetch_func;
                let io_runtime = polars_io::pl_async::get_runtime();
                let current_path_index = self.current_path_index;
                let current_max_row_group_height = self.current_max_row_group_height;

                // Push calculation of byte ranges to a task to run in parallel, as it can be
                // expensive for very wide tables and projections.
                let handle = async_executor::spawn(TaskPriority::Low, async move {
                    let byte_source = if let DynByteSource::MemSlice(mem_slice) =
                        current_byte_source.as_ref()
                    {
                        // Skip byte range calculation for `no_prefetch`.
                        if memory_prefetch_func as usize != mem_prefetch_funcs::no_prefetch as usize
                        {
                            let slice = mem_slice.0.as_ref();

                            if let Some(columns) = projection.as_ref() {
                                for range in get_row_group_byte_ranges_for_projection(
                                    &row_group_metadata,
                                    columns.as_ref(),
                                ) {
                                    memory_prefetch_func(unsafe {
                                        slice.get_unchecked_release(range)
                                    })
                                }
                            } else {
                                let mut iter = get_row_group_byte_ranges(&row_group_metadata);
                                let first = iter.next().unwrap();
                                let range =
                                    iter.fold(first, |l, r| l.start.min(r.start)..l.end.max(r.end));

                                memory_prefetch_func(unsafe { slice.get_unchecked_release(range) })
                            };
                        }

                        // We have a mmapped or in-memory slice representing the entire
                        // file that can be sliced directly, so we can skip the byte-range
                        // calculations and HashMap allocation.
                        let mem_slice = mem_slice.0.clone();
                        FetchedBytes::MemSlice {
                            offset: 0,
                            mem_slice,
                        }
                    } else if let Some(columns) = projection.as_ref() {
                        let ranges = get_row_group_byte_ranges_for_projection(
                            &row_group_metadata,
                            columns.as_ref(),
                        )
                        .collect::<Arc<[_]>>();

                        let bytes = {
                            let ranges_2 = ranges.clone();
                            task_handles_ext::AbortOnDropHandle(io_runtime.spawn(async move {
                                current_byte_source.get_ranges(ranges_2.as_ref()).await
                            }))
                            .await
                            .unwrap()?
                        };

                        assert_eq!(bytes.len(), ranges.len());

                        let mut bytes_map = PlHashMap::with_capacity(ranges.len());

                        for (range, bytes) in ranges.iter().zip(bytes) {
                            memory_prefetch_func(bytes.as_ref());
                            let v = bytes_map.insert(range.start, bytes);
                            debug_assert!(v.is_none(), "duplicate range start {}", range.start);
                        }

                        FetchedBytes::BytesMap(bytes_map)
                    } else {
                        // We have a dedicated code-path for a full projection that performs a
                        // single range request for the entire row group. During testing this
                        // provided much higher throughput from cloud than making multiple range
                        // request with `get_ranges()`.
                        let mut iter = get_row_group_byte_ranges(&row_group_metadata);
                        let mut ranges = Vec::with_capacity(iter.len());
                        let first = iter.next().unwrap();
                        ranges.push(first.clone());
                        let full_range = iter.fold(first, |l, r| {
                            ranges.push(r.clone());
                            l.start.min(r.start)..l.end.max(r.end)
                        });

                        let mem_slice = {
                            let full_range_2 = full_range.clone();
                            task_handles_ext::AbortOnDropHandle(io_runtime.spawn(async move {
                                current_byte_source.get_range(full_range_2).await
                            }))
                            .await
                            .unwrap()?
                        };

                        FetchedBytes::MemSlice {
                            offset: full_range.start,
                            mem_slice,
                        }
                    };

                    PolarsResult::Ok(RowGroupData {
                        byte_source,
                        path_index: current_path_index,
                        row_offset: current_row_offset,
                        slice,
                        file_max_row_group_height: current_max_row_group_height,
                        row_group_metadata,
                        shared_file_state: current_shared_file_state.clone(),
                    })
                });

                let handle = async_executor::AbortOnDropHandle::new(handle);
                return Some(Ok(handle));
            }

            // Initialize state to the next file.
            if !self.init_next_file_state().await {
                break;
            }
        }

        None
    }
}

enum FetchedBytes {
    MemSlice { mem_slice: MemSlice, offset: usize },
    BytesMap(PlHashMap<usize, MemSlice>),
}

impl FetchedBytes {
    fn get_range(&self, range: std::ops::Range<usize>) -> MemSlice {
        match self {
            Self::MemSlice { mem_slice, offset } => {
                let offset = *offset;
                debug_assert!(range.start >= offset);
                mem_slice.slice(range.start - offset..range.end - offset)
            },
            Self::BytesMap(v) => {
                let v = v.get(&range.start).unwrap();
                debug_assert_eq!(v.len(), range.len());
                v.clone()
            },
        }
    }
}

#[rustfmt::skip]
type RowGroupDataStreamFut = std::pin::Pin<Box<
    dyn Future<
        Output =
            (
                Box<RowGroupDataFetcher>           ,
                Option                             <
                PolarsResult                       <
                async_executor::AbortOnDropHandle  <
                PolarsResult                       <
                RowGroupData     >     >     >     >
            )
    > + Send
>>;

struct RowGroupDataStream {
    current_future: RowGroupDataStreamFut,
}

impl RowGroupDataStream {
    fn new(row_group_data_fetcher: RowGroupDataFetcher) -> Self {
        // [`RowGroupDataFetcher`] is a big struct, so we Box it once here to avoid boxing it on
        // every `next()` call.
        let current_future = Self::call_next_owned(Box::new(row_group_data_fetcher));
        Self { current_future }
    }

    fn call_next_owned(
        mut row_group_data_fetcher: Box<RowGroupDataFetcher>,
    ) -> RowGroupDataStreamFut {
        Box::pin(async move {
            let out = row_group_data_fetcher.next().await;
            (row_group_data_fetcher, out)
        })
    }
}

impl futures::stream::Stream for RowGroupDataStream {
    type Item = PolarsResult<async_executor::AbortOnDropHandle<PolarsResult<RowGroupData>>>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::pin::Pin;
        use std::task::Poll;

        match Pin::new(&mut self.current_future.as_mut()).poll(cx) {
            Poll::Ready((row_group_data_fetcher, out)) => {
                if out.is_some() {
                    self.current_future = Self::call_next_owned(row_group_data_fetcher);
                }

                Poll::Ready(out)
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

/// State shared across row groups for a single file.
struct SharedFileState {
    path_index: usize,
    hive_series: Vec<Series>,
    file_path_series: Option<Series>,
}

/// Turns row group data into DataFrames.
struct RowGroupDecoder {
    paths: Arc<Vec<PathBuf>>,
    hive_partitions: Option<Arc<Vec<HivePartitions>>>,
    hive_partitions_width: usize,
    include_file_paths: Option<Arc<str>>,
    projected_arrow_fields: Arc<[polars_core::prelude::ArrowField]>,
    row_index: Option<RowIndex>,
    physical_predicate: Option<Arc<dyn PhysicalIoExpr>>,
    ideal_morsel_size: usize,
}

impl RowGroupDecoder {
    async fn row_group_data_to_df(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<Vec<DataFrame>> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize
            + self.projected_arrow_fields.len()
            + self.hive_partitions_width
            + self.include_file_paths.is_some() as usize;

        let mut out_columns = Vec::with_capacity(out_width);

        if self.row_index.is_some() {
            // Add a placeholder so that we don't have to shift the entire vec
            // later.
            out_columns.push(Series::default());
        }

        let slice_range = row_group_data
            .slice
            .map(|(offset, len)| offset..offset + len)
            .unwrap_or(0..row_group_data.row_group_metadata.num_rows());

        let projected_arrow_fields = &self.projected_arrow_fields;
        let projected_arrow_fields = projected_arrow_fields.clone();

        let row_group_data_2 = row_group_data.clone();
        let slice_range_2 = slice_range.clone();

        // Minimum number of values to amortize the overhead of spawning tasks.
        // This value is arbitrarily chosen.
        const VALUES_PER_THREAD: usize = 16_777_216;
        let n_rows = row_group_data.row_group_metadata.num_rows();
        let cols_per_task = 1 + VALUES_PER_THREAD / n_rows;

        let decode_fut_iter = (0..self.projected_arrow_fields.len())
            .step_by(cols_per_task)
            .map(move |offset| {
                let row_group_data = row_group_data_2.clone();
                let slice_range = slice_range_2.clone();
                let projected_arrow_fields = projected_arrow_fields.clone();

                async move {
                    (offset
                        ..offset
                            .saturating_add(cols_per_task)
                            .min(projected_arrow_fields.len()))
                        .map(|i| {
                            let arrow_field = projected_arrow_fields[i].clone();

                            let columns_to_deserialize = row_group_data
                                .row_group_metadata
                                .columns()
                                .iter()
                                .filter(|col_md| {
                                    col_md.descriptor().path_in_schema[0] == arrow_field.name
                                })
                                .map(|col_md| {
                                    let (offset, len) = col_md.byte_range();
                                    let offset = offset as usize;
                                    let len = len as usize;

                                    (
                                        col_md,
                                        row_group_data.byte_source.get_range(offset..offset + len),
                                    )
                                })
                                .collect::<Vec<_>>();

                            assert!(
                                slice_range.end <= row_group_data.row_group_metadata.num_rows()
                            );

                            let array = polars_io::prelude::_internal::to_deserializer(
                                columns_to_deserialize,
                                arrow_field.clone(),
                                Some(polars_parquet::read::Filter::Range(slice_range.clone())),
                            )?;

                            let series = Series::try_from((&arrow_field, array))?;

                            // TODO: Also load in the metadata.

                            PolarsResult::Ok(series)
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                }
            });

        if decode_fut_iter.len() > 1 {
            for handle in decode_fut_iter.map(|fut| {
                async_executor::AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    fut,
                ))
            }) {
                out_columns.extend(handle.await?);
            }
        } else {
            for fut in decode_fut_iter {
                out_columns.extend(fut.await?);
            }
        }

        let projection_height = if self.projected_arrow_fields.is_empty() {
            slice_range.len()
        } else {
            debug_assert!(out_columns.len() > self.row_index.is_some() as usize);
            out_columns.last().unwrap().len()
        };

        if let Some(RowIndex { name, offset }) = self.row_index.as_ref() {
            let Some(offset) = (|| {
                let offset = offset
                    .checked_add((row_group_data.row_offset + slice_range.start) as IdxSize)?;
                offset.checked_add(projection_height as IdxSize)?;

                Some(offset)
            })() else {
                let msg = format!(
                    "adding a row index column with offset {} overflows at {} rows",
                    offset,
                    row_group_data.row_offset + slice_range.end
                );
                polars_bail!(ComputeError: msg)
            };

            // The DataFrame can be empty at this point if no columns were projected from the file,
            // so we create the row index column manually instead of using `df.with_row_index` to
            // ensure it has the correct number of rows.
            let mut ca = IdxCa::from_vec(
                name,
                (offset..offset + projection_height as IdxSize).collect(),
            );
            ca.set_sorted_flag(IsSorted::Ascending);

            out_columns[0] = ca.into_series();
        }

        let shared_file_state = row_group_data
            .shared_file_state
            .get_or_init(|| async {
                let path_index = row_group_data.path_index;

                let hive_series = if let Some(hp) = self.hive_partitions.as_deref() {
                    let mut v = hp[path_index].materialize_partition_columns();
                    for s in v.iter_mut() {
                        *s = s.new_from_index(0, row_group_data.file_max_row_group_height);
                    }
                    v
                } else {
                    vec![]
                };

                let file_path_series = self.include_file_paths.as_deref().map(|file_path_col| {
                    StringChunked::full(
                        file_path_col,
                        self.paths[path_index].to_str().unwrap(),
                        row_group_data.file_max_row_group_height,
                    )
                    .into_series()
                });

                SharedFileState {
                    path_index,
                    hive_series,
                    file_path_series,
                }
            })
            .await;

        assert_eq!(shared_file_state.path_index, row_group_data.path_index);

        for s in &shared_file_state.hive_series {
            debug_assert!(s.len() >= projection_height);
            out_columns.push(s.slice(0, projection_height));
        }

        if let Some(file_path_series) = &shared_file_state.file_path_series {
            debug_assert!(file_path_series.len() >= projection_height);
            out_columns.push(file_path_series.slice(0, projection_height));
        }

        let df = unsafe { DataFrame::new_no_checks(out_columns) };

        // Re-calculate: A slice may have been applied.
        let cols_per_task = 1 + VALUES_PER_THREAD / df.height();

        let df = if let Some(predicate) = self.physical_predicate.as_deref() {
            let mask = predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            if cols_per_task <= df.width() {
                df._filter_seq(mask)?
            } else {
                let mask = mask.clone();
                let cols = Arc::new(df.take_columns());
                let mut out_cols = Vec::with_capacity(cols.len());

                for handle in (0..cols.len())
                    .step_by(cols_per_task)
                    .map(move |offset| {
                        let cols = cols.clone();
                        let mask = mask.clone();
                        async move {
                            cols[offset..offset.saturating_add(cols_per_task).min(cols.len())]
                                .iter()
                                .map(|s| s.filter(&mask))
                                .collect::<PolarsResult<Vec<_>>>()
                        }
                    })
                    .map(|fut| {
                        async_executor::AbortOnDropHandle::new(async_executor::spawn(
                            TaskPriority::Low,
                            fut,
                        ))
                    })
                {
                    out_cols.extend(handle.await?);
                }

                unsafe { DataFrame::new_no_checks(out_cols) }
            }
        } else {
            df
        };

        assert_eq!(df.width(), out_width);

        let n_morsels = if df.height() > 3 * self.ideal_morsel_size / 2 {
            // num_rows > (1.5 * ideal_morsel_size)
            (df.height() / self.ideal_morsel_size).max(2)
        } else {
            1
        } as u64;

        if n_morsels == 1 {
            return Ok(vec![df]);
        }

        let rows_per_morsel = 1 + df.height() / n_morsels as usize;

        let out = (0..i64::try_from(df.height()).unwrap())
            .step_by(rows_per_morsel)
            .map(|offset| df.slice(offset, rows_per_morsel))
            .collect::<Vec<_>>();

        Ok(out)
    }
}

/// Read the metadata bytes of a parquet file, does not decode the bytes. If during metadata fetch
/// the bytes of the entire file are loaded, it is returned in the second return value.
async fn read_parquet_metadata_bytes(
    byte_source: &DynByteSource,
    verbose: bool,
) -> PolarsResult<(MemSlice, Option<MemSlice>)> {
    use polars_parquet::parquet::error::ParquetError;
    use polars_parquet::parquet::PARQUET_MAGIC;

    const FOOTER_HEADER_SIZE: usize = polars_parquet::parquet::FOOTER_SIZE as usize;

    let file_size = byte_source.get_size().await?;

    if file_size < FOOTER_HEADER_SIZE {
        return Err(ParquetError::OutOfSpec(format!(
            "file size ({}) is less than minimum size required to store parquet footer ({})",
            file_size, FOOTER_HEADER_SIZE
        ))
        .into());
    }

    let estimated_metadata_size = if let DynByteSource::MemSlice(_) = byte_source {
        // Mmapped or in-memory, reads are free.
        file_size
    } else {
        (file_size / 2048).clamp(16_384, 131_072).min(file_size)
    };

    let bytes = byte_source
        .get_range((file_size - estimated_metadata_size)..file_size)
        .await?;

    let footer_header_bytes = bytes.slice((bytes.len() - FOOTER_HEADER_SIZE)..bytes.len());

    let (v, remaining) = footer_header_bytes.split_at(4);
    let footer_size = i32::from_le_bytes(v.try_into().unwrap());

    if remaining != PARQUET_MAGIC {
        return Err(ParquetError::OutOfSpec(format!(
            r#"expected parquet magic bytes "{}" in footer, got "{}" instead"#,
            std::str::from_utf8(&PARQUET_MAGIC).unwrap(),
            String::from_utf8_lossy(remaining)
        ))
        .into());
    }

    if footer_size < 0 {
        return Err(ParquetError::OutOfSpec(format!(
            "expected positive footer size, got {} instead",
            footer_size
        ))
        .into());
    }

    let footer_size = footer_size as usize + FOOTER_HEADER_SIZE;

    if file_size < footer_size {
        return Err(ParquetError::OutOfSpec(format!(
            "file size ({}) is less than the indicated footer size ({})",
            file_size, footer_size
        ))
        .into());
    }

    if bytes.len() < footer_size {
        debug_assert!(!matches!(byte_source, DynByteSource::MemSlice(_)));
        if verbose {
            eprintln!(
                "[ParquetSource]: Extra {} bytes need to be fetched for metadata \
            (initial estimate = {}, actual size = {})",
                footer_size - estimated_metadata_size,
                bytes.len(),
                footer_size,
            );
        }

        let mut out = Vec::with_capacity(footer_size);
        let offset = file_size - footer_size;
        let len = footer_size - bytes.len();
        let delta_bytes = byte_source.get_range(offset..(offset + len)).await?;

        debug_assert!(out.capacity() >= delta_bytes.len() + bytes.len());

        out.extend_from_slice(&delta_bytes);
        out.extend_from_slice(&bytes);

        Ok((MemSlice::from_vec(out), None))
    } else {
        if verbose && !matches!(byte_source, DynByteSource::MemSlice(_)) {
            eprintln!(
                "[ParquetSource]: Fetched all bytes for metadata on first try \
                (initial estimate = {}, actual size = {}, excess = {})",
                bytes.len(),
                footer_size,
                estimated_metadata_size - footer_size,
            );
        }

        let metadata_bytes = bytes.slice((bytes.len() - footer_size)..bytes.len());

        if bytes.len() == file_size {
            Ok((metadata_bytes, Some(bytes)))
        } else {
            debug_assert!(!matches!(byte_source, DynByteSource::MemSlice(_)));
            let metadata_bytes = if bytes.len() - footer_size >= bytes.len() {
                // Re-allocate to drop the excess bytes
                MemSlice::from_vec(metadata_bytes.to_vec())
            } else {
                metadata_bytes
            };

            Ok((metadata_bytes, None))
        }
    }
}

fn get_row_group_byte_ranges(
    row_group_metadata: &RowGroupMetaData,
) -> impl ExactSizeIterator<Item = std::ops::Range<usize>> + '_ {
    let row_group_columns = row_group_metadata.columns();

    row_group_columns.iter().map(|rg_col_metadata| {
        let (offset, len) = rg_col_metadata.byte_range();
        (offset as usize)..(offset + len) as usize
    })
}

/// TODO: This is quadratic - incorporate https://github.com/pola-rs/polars/pull/18327 that is
/// merged.
fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetaData,
    columns: &'a [String],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    let row_group_columns = row_group_metadata.columns();

    row_group_columns.iter().filter_map(move |rg_col_metadata| {
        for col_name in columns {
            if &rg_col_metadata.descriptor().path_in_schema[0] == col_name {
                let (offset, len) = rg_col_metadata.byte_range();
                let range = (offset as usize)..((offset + len) as usize);
                return Some(range);
            }
        }
        None
    })
}

/// Ensures that a parquet file has all the necessary columns for a projection with the correct
/// dtype. There are no ordering requirements and extra columns are permitted.
fn ensure_metadata_has_projected_fields(
    projected_fields: &[polars_core::prelude::ArrowField],
    metadata: &FileMetaData,
) -> PolarsResult<()> {
    let schema = polars_parquet::arrow::read::infer_schema(metadata)?;

    // Note: We convert to Polars-native dtypes for timezone normalization.
    let mut schema = schema
        .fields
        .into_iter()
        .map(|x| {
            let dtype = DataType::from_arrow(&x.data_type, true);
            (x.name, dtype)
        })
        .collect::<PlHashMap<String, DataType>>();

    for field in projected_fields {
        let Some(dtype) = schema.remove(&field.name) else {
            polars_bail!(SchemaMismatch: "did not find column: {}", field.name)
        };

        let expected_dtype = DataType::from_arrow(&field.data_type, true);

        if dtype != expected_dtype {
            polars_bail!(SchemaMismatch: "data type mismatch for column {}: found: {}, expected: {}",
                &field.name, dtype, expected_dtype
            )
        }
    }

    Ok(())
}

fn get_memory_prefetch_func(verbose: bool) -> fn(&[u8]) -> () {
    let memory_prefetch_func = match std::env::var("POLARS_MEMORY_PREFETCH").ok().as_deref() {
        None => {
            // Sequential advice was observed to provide speedups on Linux.
            // ref https://github.com/pola-rs/polars/pull/18152#discussion_r1721701965
            #[cfg(target_os = "linux")]
            {
                mem_prefetch_funcs::madvise_sequential
            }
            #[cfg(not(target_os = "linux"))]
            {
                mem_prefetch_funcs::no_prefetch
            }
        },
        Some("no_prefetch") => mem_prefetch_funcs::no_prefetch,
        Some("prefetch_l2") => mem_prefetch_funcs::prefetch_l2,
        Some("madvise_sequential") => {
            #[cfg(target_family = "unix")]
            {
                mem_prefetch_funcs::madvise_sequential
            }
            #[cfg(not(target_family = "unix"))]
            {
                panic!("POLARS_MEMORY_PREFETCH=madvise_sequential is not supported by this system");
            }
        },
        Some("madvise_willneed") => {
            #[cfg(target_family = "unix")]
            {
                mem_prefetch_funcs::madvise_willneed
            }
            #[cfg(not(target_family = "unix"))]
            {
                panic!("POLARS_MEMORY_PREFETCH=madvise_willneed is not supported by this system");
            }
        },
        Some("madvise_populate_read") => {
            #[cfg(target_os = "linux")]
            {
                mem_prefetch_funcs::madvise_populate_read
            }
            #[cfg(not(target_os = "linux"))]
            {
                panic!(
                    "POLARS_MEMORY_PREFETCH=madvise_populate_read is not supported by this system"
                );
            }
        },
        Some(v) => panic!("invalid value for POLARS_MEMORY_PREFETCH: {}", v),
    };

    if verbose {
        let func_name = match memory_prefetch_func as usize {
            v if v == mem_prefetch_funcs::no_prefetch as usize => "no_prefetch",
            v if v == mem_prefetch_funcs::prefetch_l2 as usize => "prefetch_l2",
            v if v == mem_prefetch_funcs::madvise_sequential as usize => "madvise_sequential",
            v if v == mem_prefetch_funcs::madvise_willneed as usize => "madvise_willneed",
            v if v == mem_prefetch_funcs::madvise_populate_read as usize => "madvise_populate_read",
            _ => unreachable!(),
        };

        eprintln!("[ParquetSource] Memory prefetch function: {}", func_name);
    }

    memory_prefetch_func
}

mod mem_prefetch_funcs {
    pub use polars_utils::mem::{
        madvise_populate_read, madvise_sequential, madvise_willneed, prefetch_l2,
    };

    pub fn no_prefetch(_: &[u8]) {}
}
