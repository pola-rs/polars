use std::collections::VecDeque;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use futures::StreamExt;
use futures::stream::BoxStream;
use polars_core::prelude::{AnyValue, DataType};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy, ScanSource};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_file_reader::extra_ops::apply::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::extra_ops::missing_columns::initialize_missing_columns_policy;
use crate::nodes::io_sources::multi_file_reader::extra_ops::{
    ExtraOperations, apply_extra_columns_policy,
};
use crate::nodes::io_sources::multi_file_reader::initialization::MultiScanTaskInitializer;
use crate::nodes::io_sources::multi_file_reader::initialization::slice::{
    ResolvedSliceInfo, resolve_to_positive_slice,
};
use crate::nodes::io_sources::multi_file_reader::post_apply_pipeline::PostApplyPool;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputRecv;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks,
};

impl MultiScanTaskInitializer {
    /// Generic reader pipeline that should work for all file types and configurations
    pub async fn init_and_run(
        self,
        bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
        skip_files_mask: Option<Bitmap>,
        predicate: Option<ScanIOPredicate>,
    ) -> PolarsResult<JoinHandle<PolarsResult<()>>> {
        let verbose = self.config.verbose;
        let reader_capabilities = self.config.file_reader_builder.reader_capabilities();

        // Row index should only be pushed if we have a predicate or negative slice as there is a
        // serial synchronization cost.
        if self.config.row_index.is_some() {
            debug_assert!(
                self.config.predicate.is_some()
                    || matches!(self.config.pre_slice, Some(Slice::Negative { .. }))
            );
        }

        let ResolvedSliceInfo {
            scan_source_idx,
            row_index,
            pre_slice,
            initialized_readers,
        } = match self.config.pre_slice {
            // This can hugely benefit NDJSON, as it can read backwards.
            Some(Slice::Negative { .. })
                if self.config.sources.len() == 1
                    && reader_capabilities.contains(ReaderCapabilities::NEGATIVE_PRE_SLICE)
                    && (self.config.row_index.is_none()
                        || reader_capabilities.contains(ReaderCapabilities::ROW_INDEX)) =>
            {
                if verbose {
                    eprintln!("[MultiScanTaskInitializer]: Single file negative slice");
                }

                ResolvedSliceInfo {
                    scan_source_idx: 0,
                    row_index: self.config.row_index.clone(),
                    pre_slice: self.config.pre_slice.clone(),
                    initialized_readers: None,
                }
            },
            _ => {
                if let Some(Slice::Negative { .. }) = self.config.pre_slice {
                    if verbose {
                        eprintln!(
                            "[MultiScanTaskInitializer]: Begin resolving negative slice to positive"
                        );
                    }
                }

                resolve_to_positive_slice(&self.config).await?
            },
        };

        let cast_columns_policy = self.config.cast_columns_policy.clone();
        let missing_columns_policy = self.config.missing_columns_policy;
        let include_file_paths = self.config.include_file_paths.clone();

        let extra_ops = ExtraOperations {
            row_index,
            pre_slice,
            cast_columns_policy,
            missing_columns_policy,
            include_file_paths,
            predicate,
        };

        if verbose {
            eprintln!(
                "[MultiScanTaskInitializer]: \
                scan_source_idx: {} \
                extra_ops: {:?} \
                ",
                scan_source_idx, &extra_ops,
            )
        }

        // Pre-initialized readers if we resolved a negative slice.
        let mut initialized_readers: VecDeque<(Box<dyn FileReader>, IdxSize)> = initialized_readers
            .map(|(idx, readers)| {
                // Sanity check
                assert_eq!(idx, scan_source_idx);
                readers
            })
            .unwrap_or_default();

        let has_row_index_or_slice = extra_ops.has_row_index_or_slice();

        let config = self.config.clone();

        // Buffered initialization stream. This concurrently calls `FileReader::initialize()`,
        // allowing for e.g. concurrent Parquet metadata fetch.
        let readers_init_iter = {
            let skip_files_mask = skip_files_mask.clone();

            // If a negative slice was initialized, the length of the initialized readers will be the exact
            // stopping position.
            let end = if initialized_readers.is_empty() {
                self.config.sources.len()
            } else {
                scan_source_idx + initialized_readers.len()
            };

            let range = scan_source_idx..end;

            if verbose {
                eprintln!(
                    "\
                    [MultiScanTaskInitializer]: Readers init range: {:?} ({} / {} files)",
                    &range,
                    range.len(),
                    self.config.sources.len(),
                )
            }

            futures::stream::iter(range)
                .map(move |scan_source_idx| {
                    let cloud_options = config.cloud_options.clone();
                    let file_reader_builder = config.file_reader_builder.clone();
                    let sources = config.sources.clone();
                    let skip_files_mask = skip_files_mask.clone();

                    let maybe_initialized = initialized_readers.pop_front();
                    let scan_source = sources.get(scan_source_idx).unwrap().into_owned();

                    AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
                        let (scan_source, reader, n_rows_in_file) = async {
                            if verbose {
                                eprintln!("[MultiScan]: Initialize source {}", scan_source_idx);
                            }

                            let scan_source = scan_source?;

                            if let Some((reader, n_rows_in_file)) = maybe_initialized {
                                return PolarsResult::Ok((
                                    scan_source,
                                    reader,
                                    Some(n_rows_in_file),
                                ));
                            }

                            let mut reader = file_reader_builder.build_file_reader(
                                scan_source.clone(),
                                cloud_options,
                                scan_source_idx,
                            );

                            // Skip initialization if this file is filtered, this can save some cloud calls / metadata deserialization.
                            // Downstream must also check against `skip_files_mask` and avoid calling any functions on this reader
                            // if it is filtered out.
                            if !has_row_index_or_slice
                                && skip_files_mask.is_some_and(|x| x.get_bit(scan_source_idx))
                            {
                                return Ok((scan_source, reader, None));
                            }

                            reader.initialize().await?;
                            PolarsResult::Ok((scan_source, reader, None))
                        }
                        .await?;

                        Ok((scan_source_idx, scan_source, reader, n_rows_in_file))
                    }))
                })
                .buffered(
                    self.config
                        .n_readers_pre_init()
                        .min(self.config.sources.len()),
                )
        };

        let sources = self.config.sources.clone();
        let readers_init_iter = readers_init_iter.boxed();
        let hive_parts = self.config.hive_parts.clone();
        let final_output_schema = self.config.final_output_schema.clone();
        let projected_file_schema = self.config.projected_file_schema.clone();
        let full_file_schema = self.config.full_file_schema.clone();
        let num_pipelines = self.config.num_pipelines();
        let max_concurrent_scans = self.config.max_concurrent_scans();

        let (started_reader_tx, started_reader_rx) =
            tokio::sync::mpsc::channel(max_concurrent_scans.max(2) - 1);

        let reader_starter_handle = AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::Low,
            ReaderStarter {
                reader_capabilities,
                n_sources: sources.len(),

                readers_init_iter,
                started_reader_tx,
                max_concurrent_scans,
                skip_files_mask,
                extra_ops,
                constant_args: StartReaderArgsConstant {
                    hive_parts,
                    final_output_schema,
                    projected_file_schema,
                    missing_columns_policy: self.config.missing_columns_policy,
                    full_file_schema,
                    extra_columns_policy: self.config.extra_columns_policy,
                },
                num_pipelines,
                verbose,
            }
            .run(),
        ));

        let attach_to_bridge_handle = AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::Low,
            AttachReaderToBridge {
                started_reader_rx,
                bridge_recv_port_tx,
                num_pipelines,
                verbose,
            }
            .run(),
        ));

        let handle = async_executor::spawn(TaskPriority::Low, async move {
            attach_to_bridge_handle.await?;
            reader_starter_handle.await?;
            Ok(())
        });

        Ok(handle)
    }
}

/// Starts readers, potentially multiple at the same time if it can.
struct ReaderStarter {
    reader_capabilities: ReaderCapabilities,
    #[expect(clippy::type_complexity)]
    readers_init_iter:
        BoxStream<'static, PolarsResult<(usize, ScanSource, Box<dyn FileReader>, Option<IdxSize>)>>,
    n_sources: usize,
    started_reader_tx: tokio::sync::mpsc::Sender<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    max_concurrent_scans: usize,
    skip_files_mask: Option<Bitmap>,
    extra_ops: ExtraOperations,
    constant_args: StartReaderArgsConstant,
    num_pipelines: usize,
    verbose: bool,
}

impl ReaderStarter {
    async fn run(self) -> PolarsResult<()> {
        let ReaderStarter {
            reader_capabilities,
            mut readers_init_iter,
            n_sources,
            started_reader_tx,
            max_concurrent_scans,
            skip_files_mask,
            extra_ops,
            constant_args,
            num_pipelines,
            verbose,
        } = self;

        // Note: This is unused if we aren't slicing or row indexing.
        let mut current_row_position: IdxSize = 0;

        if !extra_ops.has_row_index_or_slice() {
            // Set to IdxSize::MAX if we expect it to be unused. This way if it is incorrectly being
            // used it should cause an error.
            current_row_position = IdxSize::MAX;
        }

        let wait_group = WaitGroup::default();

        loop {
            // Note: This loop should only do basic bookkeeping (e.g. slice position) and reader initialization.
            // It should avoid doing compute as much as possible - those should instead be deferred to spawned tasks.

            let pre_slice_this_file = extra_ops.pre_slice.clone().map(|x| match x {
                Slice::Positive { .. } => x.offsetted(current_row_position as usize),
                Slice::Negative { .. } => x,
            });

            if pre_slice_this_file.is_some() && verbose {
                eprintln!(
                    "[ReaderStarter]: current_row_position: {}, pre_slice_this_file: {:?}",
                    current_row_position,
                    pre_slice_this_file.as_ref().unwrap(),
                )
            }

            if pre_slice_this_file.as_ref().is_some_and(|x| x.len() == 0) {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (pre_slice)")
                }
                break;
            }

            let Some((scan_source_idx, scan_source, mut reader, opt_n_rows_in_file)) =
                readers_init_iter.next().await.transpose()?
            else {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (no more readers)")
                }
                break;
            };

            if verbose {
                eprintln!("[ReaderStarter]: scan_source_idx: {}", scan_source_idx)
            }

            if skip_files_mask
                .as_ref()
                .is_some_and(|x| x.get_bit(scan_source_idx))
            {
                if verbose {
                    eprintln!(
                        "[ReaderStarter]: Skip read of file index {} (skip_files_mask)",
                        scan_source_idx
                    )
                }

                if extra_ops.has_row_index_or_slice() {
                    // Should never: Negative slice should only hit this loop in the case:
                    // * Single NDJSON file that is not filtered out.
                    if let Some(Slice::Negative { .. }) = pre_slice_this_file {
                        panic!();
                    }

                    current_row_position = current_row_position.saturating_add(
                        reader.row_position_after_slice(pre_slice_this_file).await?,
                    );
                }

                continue;
            }

            let row_index_this_file = extra_ops.row_index.clone().map(|mut ri| {
                ri.offset = ri.offset.saturating_add(current_row_position);
                ri
            });

            let extra_ops_this_file = ExtraOperations {
                row_index: row_index_this_file,
                pre_slice: pre_slice_this_file.clone(),
                // Other operations don't need updating per file
                ..extra_ops.clone()
            };

            let (row_position_on_end_tx, row_position_on_end_rx) =
                if extra_ops.has_row_index_or_slice() && n_sources - scan_source_idx > 1 {
                    let (mut tx, rx) = connector::connector();

                    // See if we have the value leftover from negative slice initialization, so we don't duplicate row counting.
                    if let Some(mut n_rows) = opt_n_rows_in_file {
                        if let Some(pre_slice) = pre_slice_this_file {
                            n_rows = IdxSize::try_from(
                                pre_slice
                                    .restrict_to_bounds(usize::try_from(n_rows).unwrap())
                                    .end_position(),
                            )
                            .unwrap_or(IdxSize::MAX);
                        }

                        _ = tx.try_send(n_rows);
                        (None, Some(rx))
                    } else {
                        (Some(tx), Some(rx))
                    }
                } else {
                    (None, None)
                };

            // Note: If a reader does not support this we can also have the post apply pipeline do
            // this callback for us (but it will be slightly slower).
            let callbacks = FileReaderCallbacks {
                row_position_on_end_tx,
                ..Default::default()
            };

            let mut extra_ops_post = extra_ops_this_file;

            let row_index = if reader_capabilities.contains(ReaderCapabilities::ROW_INDEX) {
                extra_ops_post.row_index.take()
            } else {
                None
            };

            let pre_slice = match &extra_ops_post.pre_slice {
                Some(Slice::Positive { .. })
                    if reader_capabilities.contains(ReaderCapabilities::PRE_SLICE) =>
                {
                    extra_ops_post.pre_slice.take()
                },
                Some(Slice::Negative { .. })
                    if reader_capabilities.contains(ReaderCapabilities::NEGATIVE_PRE_SLICE) =>
                {
                    extra_ops_post.pre_slice.take()
                },
                _ => None,
            };

            // Note: We do set_external_columns later below to avoid blocking this loop.
            let predicate = if extra_ops_post.predicate.is_some()
                // TODO: Support cast columns in parquet
                && extra_ops_post.cast_columns_policy == CastColumnsPolicy::ERROR_ON_MISMATCH
                && reader_capabilities.contains(ReaderCapabilities::PARTIAL_FILTER)
                && extra_ops_post.row_index.is_none()
                && extra_ops_post.pre_slice.is_none()
            {
                if reader_capabilities.contains(ReaderCapabilities::FULL_FILTER) {
                    // If the reader can fully handle the predicate itself, let it do it itself.
                    extra_ops_post.predicate.take()
                } else {
                    // Otherwise, we want to pass it and filter again afterwards.
                    extra_ops_post.predicate.clone()
                }
            } else {
                None
            };

            let begin_read_args = BeginReadArgs {
                projected_schema: constant_args.projected_file_schema.clone(),
                row_index,
                pre_slice,
                predicate,
                cast_columns_policy: extra_ops_post.cast_columns_policy.clone(),
                num_pipelines,
                callbacks,
            };

            let start_args_this_file = StartReaderArgsPerFile {
                scan_source,
                scan_source_idx,
                reader,
                begin_read_args,
                extra_ops_post,
            };

            let reader_start_task_handle = AbortOnDropHandle::new(async_executor::spawn(
                TaskPriority::Low,
                start_reader_impl(constant_args.clone(), start_args_this_file),
            ));

            if started_reader_tx
                .send((reader_start_task_handle, wait_group.token()))
                .await
                .is_err()
            {
                break;
            };

            // If we have row index or slice, we must wait for the row position callback before
            // we can start the next reader. This will be very fast for e.g. Parquet / IPC, but
            // for CSV / NDJSON this will be slower.
            //
            // Note: If this reader ends early due to an error, we may start the next reader with an incorrect
            // row position. But downstream will never connect the next reader to the bridge as it should join
            // on this reader and already exit from the error.
            //
            // TODO:
            // * Parallelize the CSV row count
            // * NDJSON skips rows (i.e. non-zero offset) in a single-threaded manner.
            if let Some(mut rx) = row_position_on_end_rx {
                if let Ok(n) = rx.recv().await {
                    current_row_position = current_row_position.saturating_add(n);
                }
            }

            if max_concurrent_scans == 1 {
                if verbose {
                    eprintln!("[ReaderStarter]: max_concurrent_scans is 1, waiting..")
                }

                wait_group.wait().await;
            }
        }

        Ok(())
    }
}

/// Constant over the file list.
#[derive(Clone)]
struct StartReaderArgsConstant {
    hive_parts: Option<Arc<HivePartitionsDf>>,
    final_output_schema: SchemaRef,
    projected_file_schema: SchemaRef,
    missing_columns_policy: MissingColumnsPolicy,
    full_file_schema: SchemaRef,
    extra_columns_policy: ExtraColumnsPolicy,
}

struct StartReaderArgsPerFile {
    scan_source: ScanSource,
    scan_source_idx: usize,
    reader: Box<dyn FileReader>,
    begin_read_args: BeginReadArgs,
    extra_ops_post: ExtraOperations,
}

async fn start_reader_impl(
    constant_args: StartReaderArgsConstant,
    args_this_file: StartReaderArgsPerFile,
) -> PolarsResult<StartedReaderState> {
    let StartReaderArgsConstant {
        hive_parts,
        final_output_schema,
        projected_file_schema,
        missing_columns_policy,
        full_file_schema,
        extra_columns_policy,
    } = constant_args;

    let StartReaderArgsPerFile {
        scan_source,
        scan_source_idx,
        mut reader,
        mut begin_read_args,
        extra_ops_post,
    } = args_this_file;

    let pre_slice_to_reader = begin_read_args.pre_slice.clone();

    let file_schema_rx = if !matches!(extra_columns_policy, ExtraColumnsPolicy::Ignore) {
        // Upstream should not have any reason to attach this.
        assert!(begin_read_args.callbacks.file_schema_tx.is_none());
        let (tx, rx) = connector::connector();
        begin_read_args.callbacks.file_schema_tx = Some(tx);
        Some(rx)
    } else {
        None
    };

    let mut _file_schema: Option<SchemaRef> = None;

    macro_rules! get_file_schema {
        () => {{
            if _file_schema.is_none() {
                _file_schema = Some(reader.file_schema().await?)
            }

            _file_schema.clone().unwrap()
        }};
    }

    // Should not have both of these set, as the `n_rows_in_file` will cause the `row_position_on_end`
    // callback to be unnecessarily blocked in CSV and NDJSON.
    debug_assert!(
        !(begin_read_args.callbacks.row_position_on_end_tx.is_some()
            && begin_read_args.callbacks.n_rows_in_file_tx.is_some()),
    );

    if let Some(predicate) = begin_read_args.predicate.as_mut() {
        let mut external_predicate_cols = Vec::with_capacity(
            hive_parts.as_ref().map_or(0, |x| x.df().width())
                + extra_ops_post.include_file_paths.is_some() as usize,
        );

        if let Some(hp) = &hive_parts {
            external_predicate_cols.extend(
                hp.df()
                    .get_columns()
                    .iter()
                    .filter(|c| predicate.live_columns.contains(c.name()))
                    .map(|c| {
                        (
                            c.name().clone(),
                            Scalar::new(
                                c.dtype().clone(),
                                c.get(scan_source_idx).unwrap().into_static(),
                            ),
                        )
                    }),
            );
        }

        if let Some(col_name) = extra_ops_post.include_file_paths.clone() {
            external_predicate_cols.push((
                col_name,
                Scalar::new(
                    DataType::String,
                    AnyValue::StringOwned(
                        scan_source
                            .as_scan_source_ref()
                            .to_include_path_name()
                            .into(),
                    ),
                ),
            ))
        }

        let mut extra_cols = vec![];
        initialize_missing_columns_policy(
            &missing_columns_policy,
            &projected_file_schema,
            get_file_schema!().as_ref(),
            &mut extra_cols,
        )?;
        external_predicate_cols.extend(
            extra_cols
                .into_iter()
                .map(|c| (c.name().clone(), c.scalar().clone())),
        );

        predicate.set_external_constant_columns(external_predicate_cols);
    }

    let (mut reader_output_port, reader_handle) = reader.begin_read(begin_read_args)?;

    let reader_handle = AbortOnDropHandle::new(reader_handle);

    if !matches!(extra_columns_policy, ExtraColumnsPolicy::Ignore) {
        if let Ok(this_file_schema) = file_schema_rx.unwrap().recv().await {
            apply_extra_columns_policy(&extra_columns_policy, full_file_schema, this_file_schema)?;
        } else {
            drop(reader_output_port);
            return Err(reader_handle.await.unwrap_err());
        }
    }

    let first_morsel = reader_output_port.recv().await.ok();

    let ops_applier = if let Some(morsel) = first_morsel.as_ref() {
        let final_output_schema = final_output_schema.clone();
        let projected_file_schema = projected_file_schema.clone();
        let mut extra_ops = extra_ops_post;

        // The offset of the row index sent to the post apply pipeline should be the row position of
        // the first morsel sent by the reader.
        if let Some(ri) = extra_ops.row_index.as_mut() {
            let offset_by = pre_slice_to_reader.as_ref().map_or(0, |x| {
                let Slice::Positive { offset, .. } = x else {
                    unreachable!()
                };
                IdxSize::try_from(*offset).unwrap_or(IdxSize::MAX)
            });

            ri.offset = ri.offset.saturating_add(offset_by);
        }

        ApplyExtraOps::Uninitialized {
            final_output_schema,
            projected_file_schema,
            extra_ops,
            scan_source: scan_source.clone(),
            scan_source_idx,
            hive_parts,
        }
        .initialize(morsel.df().schema())?
    } else {
        ApplyExtraOps::Noop
    };

    let state = StartedReaderState {
        reader_output_port,
        first_morsel,
        ops_applier,
        reader_handle,
    };

    Ok(state)
}

/// State for a reader that has been started.
struct StartedReaderState {
    reader_output_port: FileReaderOutputRecv,
    first_morsel: Option<Morsel>,
    ops_applier: ApplyExtraOps,
    reader_handle: AbortOnDropHandle<PolarsResult<()>>,
}

struct AttachReaderToBridge {
    started_reader_rx: tokio::sync::mpsc::Receiver<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
    num_pipelines: usize,
    verbose: bool,
}

impl AttachReaderToBridge {
    async fn run(self) -> PolarsResult<()> {
        let AttachReaderToBridge {
            mut started_reader_rx,
            mut bridge_recv_port_tx,
            num_pipelines,
            verbose,
        } = self;

        let mut n_readers_received: usize = 0;

        let mut post_apply_pool: Option<PostApplyPool> = None;

        while let Some((init_task_handle, wait_token)) = started_reader_rx.recv().await {
            n_readers_received = n_readers_received.saturating_add(1);

            if verbose {
                eprintln!(
                    "[AttachReaderToBridge]: got reader, n_readers_received: {}",
                    n_readers_received
                );
            }

            let StartedReaderState {
                reader_output_port,
                first_morsel,
                ops_applier,
                reader_handle,
            } = init_task_handle.await?;

            if let Some(first_morsel) = first_morsel {
                match ops_applier {
                    ApplyExtraOps::Noop => {
                        if verbose {
                            eprintln!("[AttachReaderToBridge]: ApplyExtraOps::Noop");
                        }

                        if bridge_recv_port_tx
                            .send(BridgeRecvPort::Direct {
                                rx: reader_output_port,
                                first_morsel: Some(first_morsel),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    },

                    ApplyExtraOps::Initialized { .. } => {
                        if verbose {
                            eprintln!("[AttachReaderToBridge]: ApplyExtraOps::Initialized");
                        }

                        let post_apply_pool = post_apply_pool
                            .get_or_insert_with(|| PostApplyPool::new(num_pipelines));

                        let bridge_recv_port = post_apply_pool
                            .run_with_reader(
                                reader_output_port,
                                Arc::new(ops_applier),
                                first_morsel,
                            )
                            .await?;

                        if bridge_recv_port_tx.send(bridge_recv_port).await.is_err() {
                            break;
                        }

                        post_apply_pool.wait_current_reader().await?;
                    },

                    ApplyExtraOps::Uninitialized { .. } => unreachable!(),
                }
            }

            drop(wait_token);
            reader_handle.await?;
        }

        // Catch errors
        if let Some(post_apply_pool) = post_apply_pool {
            post_apply_pool.shutdown().await?;
        }

        Ok(())
    }
}
