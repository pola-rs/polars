use std::collections::VecDeque;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use futures::StreamExt;
use futures::stream::BoxStream;
use polars_core::prelude::{AnyValue, DataType, PlHashMap};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_core::schema::iceberg::IcebergSchema;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy, ScanSource};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::nodes::io_sources::multi_file_reader::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_file_reader::extra_ops::apply::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::extra_ops::{
    ExtraOperations, ForbidExtraColumns, missing_column_err,
};
use crate::nodes::io_sources::multi_file_reader::initialization::MultiScanTaskInitializer;
use crate::nodes::io_sources::multi_file_reader::initialization::deletion_files::{
    DeletionFilesProvider, ExternalFilterMask, RowDeletionsInit,
};
use crate::nodes::io_sources::multi_file_reader::initialization::projection::ProjectionBuilder;
use crate::nodes::io_sources::multi_file_reader::initialization::slice::{
    ResolvedSliceInfo, resolve_to_positive_slice,
};
use crate::nodes::io_sources::multi_file_reader::post_apply_pipeline::PostApplyPipeline;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};
use crate::nodes::io_sources::multi_file_reader::row_counter::RowCounter;

impl MultiScanTaskInitializer {
    /// Generic reader pipeline that should work for all file types and configurations
    pub async fn init_and_run(
        self,
        bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
        skip_files_mask: Option<Bitmap>,
        predicate: Option<ScanIOPredicate>,
    ) -> PolarsResult<JoinHandle<PolarsResult<()>>> {
        let verbose = self.config.verbose;
        let num_pipelines = self.config.num_pipelines();
        let reader_capabilities = self.config.reader_capabilities();

        // Row index should only be pushed if we have a predicate or negative slice as there is a
        // serial synchronization cost from needing to track the row position.
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
            row_deletions,
        } = match self.config.pre_slice {
            // This can hugely benefit NDJSON, as it can read backwards.
            Some(Slice::Negative { .. })
                if self.config.sources.len() == 1
                    && reader_capabilities.contains(ReaderCapabilities::NEGATIVE_PRE_SLICE)
                    && (self.config.row_index.is_none()
                        || reader_capabilities.contains(ReaderCapabilities::ROW_INDEX))
                    && (self.config.deletion_files.is_none()
                        || reader_capabilities
                            .contains(ReaderCapabilities::EXTERNAL_FILTER_MASK)) =>
            {
                if verbose {
                    eprintln!("[MultiScanTaskInitializer]: Single file negative slice");
                }

                ResolvedSliceInfo {
                    scan_source_idx: 0,
                    row_index: self.config.row_index.clone(),
                    pre_slice: self.config.pre_slice.clone(),
                    initialized_readers: None,
                    row_deletions: Default::default(),
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

        let initialized_row_deletions: Arc<PlHashMap<usize, ExternalFilterMask>> =
            Arc::new(row_deletions);

        let cast_columns_policy = self.config.cast_columns_policy.clone();
        let missing_columns_policy = self.config.missing_columns_policy;
        let include_file_paths = self.config.include_file_paths.clone();

        let extra_ops = ExtraOperations {
            row_index,
            row_index_col_idx: self.config.row_index.as_ref().map_or(usize::MAX, |x| {
                self.config.final_output_schema.index_of(&x.name).unwrap()
            }),
            pre_slice,
            include_file_paths,
            file_path_col_idx: self
                .config
                .include_file_paths
                .as_ref()
                .map_or(usize::MAX, |x| {
                    self.config.final_output_schema.index_of(x).unwrap()
                }),
            predicate,
        };

        if verbose {
            eprintln!(
                "[MultiScanTaskInitializer]: \
                scan_source_idx: {}, \
                extra_ops: {:?} \
                ",
                scan_source_idx, &extra_ops,
            )
        }

        // Pre-initialized readers if we resolved a negative slice.
        let mut initialized_readers: VecDeque<(Box<dyn FileReader>, RowCounter)> =
            initialized_readers
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

            let mut range = {
                // If a negative slice was initialized, the length of the initialized readers will be the exact
                // stopping position.
                let end = if initialized_readers.is_empty() {
                    self.config.sources.len()
                } else {
                    scan_source_idx + initialized_readers.len()
                };

                scan_source_idx..end
            };

            if verbose {
                let n_filtered = skip_files_mask
                    .clone()
                    .map_or(0, |x| x.sliced(range.start, range.len()).set_bits());
                let n_readers_init = range.len() - n_filtered;

                eprintln!(
                    "\
                    [MultiScanTaskInitializer]: Readers init: {} / ({} total) \
                    (range: {:?}, filtered out: {})",
                    n_readers_init,
                    self.config.sources.len(),
                    &range,
                    n_filtered,
                )
            }

            if let Some(skip_files_mask) = &skip_files_mask {
                range.end = range
                    .end
                    .min(skip_files_mask.len() - skip_files_mask.trailing_ones());
            }

            let range = range.filter(move |scan_source_idx| {
                let can_skip = !has_row_index_or_slice
                    && skip_files_mask
                        .as_ref()
                        .is_some_and(|x| x.get_bit(*scan_source_idx));

                !can_skip
            });

            let deletion_files_provider =
                DeletionFilesProvider::new(self.config.deletion_files.clone());

            futures::stream::iter(range)
                .map(move |scan_source_idx| {
                    let cloud_options = config.cloud_options.clone();
                    let file_reader_builder = config.file_reader_builder.clone();
                    let sources = config.sources.clone();
                    let deletion_files_provider = deletion_files_provider.clone();
                    let initialized_row_deletions = initialized_row_deletions.clone();

                    let maybe_initialized = initialized_readers.pop_front();
                    let scan_source = sources.get(scan_source_idx).unwrap().into_owned();

                    AbortOnDropHandle::new(async_executor::spawn(TaskPriority::Low, async move {
                        let (scan_source, reader, n_rows_in_file) = async {
                            if verbose {
                                eprintln!("[MultiScan]: Initialize source {scan_source_idx}");
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
                                cloud_options.clone(),
                                scan_source_idx,
                            );

                            reader.initialize().await?;
                            let opt_n_rows = reader
                                .fast_n_rows_in_file()
                                .await?
                                .map(|num_phys_rows| RowCounter::new(num_phys_rows, 0));

                            PolarsResult::Ok((scan_source, reader, opt_n_rows))
                        }
                        .await?;

                        let row_deletions: Option<RowDeletionsInit> = initialized_row_deletions
                            .get(&scan_source_idx)
                            .map(|x| RowDeletionsInit::Initialized(x.clone()))
                            .or_else(|| {
                                deletion_files_provider.spawn_row_deletions_init(
                                    scan_source_idx,
                                    cloud_options,
                                    num_pipelines,
                                    verbose,
                                )
                            });

                        Ok(InitializedReaderState {
                            scan_source_idx,
                            scan_source,
                            reader,
                            n_rows_in_file,
                            row_deletions,
                        })
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
        let file_projection_builder = self.config.file_projection_builder.clone();
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
                    reader_capabilities,
                    file_projection_builder,
                    cast_columns_policy,
                    missing_columns_policy,
                    forbid_extra_columns: self.config.forbid_extra_columns.clone(),
                    num_pipelines,
                    verbose,
                },
                verbose,
            }
            .run(),
        ));

        let attach_to_bridge_handle = AbortOnDropHandle::new(async_executor::spawn(
            TaskPriority::Low,
            AttachReaderToBridge {
                started_reader_rx,
                bridge_recv_port_tx,
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
    readers_init_iter: BoxStream<'static, PolarsResult<InitializedReaderState>>,
    n_sources: usize,
    started_reader_tx: tokio::sync::mpsc::Sender<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    max_concurrent_scans: usize,
    skip_files_mask: Option<Bitmap>,
    extra_ops: ExtraOperations,
    constant_args: StartReaderArgsConstant,
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
            verbose,
        } = self;

        // Notes:
        // * This is unused if we aren't slicing or row indexing.
        let mut current_row_position: Option<RowCounter> = extra_ops
            .has_row_index_or_slice()
            .then_some(RowCounter::default());

        let wait_group = WaitGroup::default();

        loop {
            // Note: This loop should only do basic bookkeeping (e.g. slice position) and reader initialization.
            // It should avoid doing compute as much as possible - those should instead be deferred to spawned tasks.

            let pre_slice_this_file = extra_ops
                .pre_slice
                .clone()
                .map(|x| {
                    PolarsResult::Ok(match x {
                        Slice::Positive { .. } => {
                            x.offsetted(current_row_position.unwrap().num_rows()?)
                        },
                        Slice::Negative { .. } => x,
                    })
                })
                .transpose()?;

            if current_row_position.is_some() && verbose {
                eprintln!(
                    "[ReaderStarter]: \
                    current_row_position: {current_row_position:?}, \
                    pre_slice_this_file: {pre_slice_this_file:?}"
                )
            }

            if pre_slice_this_file.as_ref().is_some_and(|x| x.len() == 0) {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (pre_slice)")
                }
                break;
            }

            let Some(InitializedReaderState {
                scan_source_idx,
                scan_source,
                mut reader,
                mut n_rows_in_file,
                row_deletions,
            }) = readers_init_iter.next().await.transpose()?
            else {
                if verbose {
                    eprintln!("[ReaderStarter]: Stopping (no more readers)")
                }
                break;
            };

            if verbose {
                eprintln!("[ReaderStarter]: scan_source_idx: {scan_source_idx}")
            }

            // Note: We `.await` here for the row deletions to be fully loaded.
            //       For this reason it's important that we already spawn background tasks to fully
            //       load them at the reader pre-initialization stage.
            let external_filter_mask = if let Some(row_deletions) = row_deletions {
                Some(row_deletions.into_external_filter_mask().await?)
            } else {
                None
            };

            // * This translates the `pre_slice` to physical offsets (i.e. apply before deleting rows).
            //   The slice remains the same if there are no row deletions.
            let pre_slice_this_file: Option<PhysicalSlice> =
                pre_slice_this_file.map(|pre_slice| match pre_slice {
                    Slice::Positive { .. } => {
                        PhysicalSlice::new(pre_slice, external_filter_mask.as_ref())
                    },

                    // This is hit here for NDJSON single file negative slice, we just passthrough
                    // in this case.
                    Slice::Negative { .. } => {
                        if external_filter_mask.is_some() {
                            unimplemented!(
                                "{pre_slice:?} {}",
                                ExternalFilterMask::log_display(external_filter_mask.as_ref())
                            )
                        }

                        assert!(
                            (extra_ops.row_index.is_none()
                                || reader_capabilities.contains(ReaderCapabilities::ROW_INDEX))
                                && (external_filter_mask.is_none()
                                    || reader_capabilities
                                        .contains(ReaderCapabilities::EXTERNAL_FILTER_MASK))
                        );

                        PhysicalSlice {
                            slice: pre_slice,
                            // Note, this is not the correct starting position. The assertion above
                            // should ensure this value is not used in post-apply.
                            slice_start_position: RowCounter::default(),
                        }
                    },
                });

            let row_index_this_file = {
                let current_row_position = if let Some(current_row_position) = current_row_position
                {
                    current_row_position.num_rows_idxsize_saturating()?
                } else {
                    IdxSize::MAX
                };

                extra_ops.row_index.clone().map(|mut ri| {
                    ri.offset = ri.offset.saturating_add(current_row_position);
                    ri
                })
            };

            let extra_ops_this_file = ExtraOperations {
                row_index: row_index_this_file,
                pre_slice: pre_slice_this_file
                    .as_ref()
                    .map(|phys_slice: &PhysicalSlice| phys_slice.slice.clone()),
                // Other operations don't need updating per file
                ..extra_ops.clone()
            };

            // &str that holds the reason
            let mut skip_read_reason: Option<&'static str> = skip_files_mask
                .as_ref()
                .is_some_and(|x| x.get_bit(scan_source_idx))
                .then_some("skip_files_mask");

            if skip_read_reason.is_some() {
                // If this is not the case then the reader does not need to be sent here.
                debug_assert!(extra_ops.has_row_index_or_slice())
            }

            // `fast_n_rows_in_file()` or negative slice, we know the exact row count here already.
            // After this point, if n_rows_in_file is `Some`, it should contain the exact physical
            // and deleted row counts.
            if let Some(n_rows_in_file) = n_rows_in_file.as_mut() {
                if let Some(external_filter_mask) = external_filter_mask.as_ref() {
                    unsafe {
                        n_rows_in_file.set_deleted_rows(external_filter_mask.num_deleted_rows())
                    }
                }

                if n_rows_in_file.num_rows()? == 0 {
                    skip_read_reason = Some("0 rows")
                } else if pre_slice_this_file.as_ref().is_some_and(|phys_slice| {
                    phys_slice
                        .slice
                        .clone()
                        .restrict_to_bounds(n_rows_in_file.num_physical_rows())
                        .len()
                        == 0
                }) {
                    skip_read_reason = Some("0 rows after slice")
                }
            }

            if let Some(skip_read_reason) = skip_read_reason {
                if verbose {
                    eprintln!(
                        "[ReaderStarter]: scan_source_idx: {scan_source_idx}: \
                        skip read ({skip_read_reason}): \
                        n_rows_in_file: {n_rows_in_file:?}, \
                        pre_slice: {pre_slice_this_file:?}"
                    )
                }

                if started_reader_tx.is_closed() {
                    break;
                }

                // We are tracking the row position so we need the row count from this file even if it's skipped.
                if extra_ops.has_row_index_or_slice() {
                    let Some(current_row_position) = current_row_position.as_mut() else {
                        panic!()
                    };

                    let pre_slice_this_file =
                        pre_slice_this_file.map(|phys_slice| phys_slice.slice);

                    // Should never: Negative slice should only hit this loop in the case:
                    // * Single NDJSON file that is not filtered out.
                    if let Some(Slice::Negative { .. }) = pre_slice_this_file {
                        panic!();
                    }

                    let get_row_count = async {
                        let num_physical_rows =
                            reader.row_position_after_slice(pre_slice_this_file).await?;

                        let num_deleted_rows = external_filter_mask.as_ref().map_or(0, |mask| {
                            mask.slice(
                                0,
                                mask.len().min(usize::try_from(num_physical_rows).unwrap()),
                            )
                            .num_deleted_rows()
                        });

                        let file_row_count = RowCounter::new(num_physical_rows, num_deleted_rows);

                        if verbose {
                            eprintln!(
                                "[ReaderStarter]: scan_source_idx: {scan_source_idx}: \
                                file_row_count: {file_row_count:?}"
                            )
                        }

                        PolarsResult::Ok(file_row_count)
                    };

                    if n_rows_in_file.is_none() {
                        n_rows_in_file = Some(get_row_count.await?)
                    } else if cfg!(debug_assertions) {
                        assert_eq!(n_rows_in_file.unwrap(), get_row_count.await?)
                    }

                    *current_row_position = current_row_position.add(n_rows_in_file.unwrap());
                }

                continue;
            }

            let (row_position_on_end_tx, row_position_on_end_rx) = if n_rows_in_file.is_none()
                && extra_ops.has_row_index_or_slice()
                && n_sources - scan_source_idx > 1
            {
                let (tx, rx) = connector::connector();
                (Some(tx), Some(rx))
            } else {
                (None, None)
            };

            let callbacks = FileReaderCallbacks {
                row_position_on_end_tx,
                ..Default::default()
            };

            let start_args_this_file = StartReaderArgsPerFile {
                scan_source,
                scan_source_idx,
                reader,
                pre_slice_this_file,
                extra_ops_this_file,
                callbacks,
                external_filter_mask: external_filter_mask.clone(),
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
            if let Some(current_row_position) = current_row_position.as_mut() {
                let mut row_position_this_file = RowCounter::default();

                #[expect(clippy::never_loop)]
                loop {
                    if let Some(v) = n_rows_in_file {
                        row_position_this_file = v;
                        break;
                    };

                    // Note, can be None on the last scan source.
                    let Some(mut rx) = row_position_on_end_rx else {
                        break;
                    };

                    let Ok(num_physical_rows) = rx.recv().await else {
                        break;
                    };

                    let num_deleted_rows = external_filter_mask.map_or(0, |external_filter_mask| {
                        external_filter_mask
                            .slice(0, usize::try_from(num_physical_rows).unwrap())
                            .num_deleted_rows()
                    });

                    row_position_this_file = RowCounter::new(num_physical_rows, num_deleted_rows);
                    break;
                }

                *current_row_position = current_row_position.add(row_position_this_file);
            }

            if skip_read_reason.is_none() && max_concurrent_scans == 1 {
                if verbose {
                    eprintln!("[ReaderStarter]: max_concurrent_scans is 1, waiting..")
                }

                wait_group.wait().await;
            }
        }

        Ok(())
    }
}

struct InitializedReaderState {
    scan_source_idx: usize,
    scan_source: ScanSource,
    reader: Box<dyn FileReader>,
    n_rows_in_file: Option<RowCounter>,
    row_deletions: Option<RowDeletionsInit>,
}

/// Constant over the file list.
#[derive(Clone)]
struct StartReaderArgsConstant {
    hive_parts: Option<Arc<HivePartitionsDf>>,
    final_output_schema: SchemaRef,
    reader_capabilities: ReaderCapabilities,
    file_projection_builder: ProjectionBuilder,
    cast_columns_policy: CastColumnsPolicy,
    missing_columns_policy: MissingColumnsPolicy,
    forbid_extra_columns: Option<ForbidExtraColumns>,
    num_pipelines: usize,
    verbose: bool,
}

struct StartReaderArgsPerFile {
    scan_source: ScanSource,
    scan_source_idx: usize,
    reader: Box<dyn FileReader>,
    pre_slice_this_file: Option<PhysicalSlice>,
    extra_ops_this_file: ExtraOperations,
    callbacks: FileReaderCallbacks,
    external_filter_mask: Option<ExternalFilterMask>,
}

/// This function gets run in a spawned task to avoid blocking the ReaderStarter's loop.
async fn start_reader_impl(
    constant_args: StartReaderArgsConstant,
    args_this_file: StartReaderArgsPerFile,
) -> PolarsResult<StartedReaderState> {
    let StartReaderArgsConstant {
        hive_parts,
        final_output_schema,
        reader_capabilities,
        file_projection_builder,
        cast_columns_policy,
        missing_columns_policy,
        forbid_extra_columns,
        num_pipelines,
        verbose,
    } = constant_args;

    let StartReaderArgsPerFile {
        scan_source,
        scan_source_idx,
        mut reader,
        pre_slice_this_file,
        extra_ops_this_file,
        mut callbacks,
        external_filter_mask,
    } = args_this_file;

    let file_iceberg_schema: Option<IcebergSchema> =
        if matches!(&file_projection_builder, ProjectionBuilder::Iceberg { .. }) {
            reader
                .file_arrow_schema()
                .await?
                .map(|x| IcebergSchema::from_arrow_schema(x.as_ref()))
                .transpose()?
        } else {
            None
        };

    let file_projection = file_projection_builder.build_projection(
        None,
        file_iceberg_schema.as_ref(),
        cast_columns_policy.clone(),
    )?;

    let mut extra_ops_post = extra_ops_this_file;

    let (
        mut projection_to_reader,
        projection_to_post,
        row_index,
        pre_slice,
        mut predicate,
        external_filter_mask,
    ) = ReaderOperationPushdown {
        file_projection: file_projection.clone(),
        reader_capabilities,
        external_filter_mask: external_filter_mask.clone(),
        extra_ops_post: &mut extra_ops_post,
    }
    .push_operations();

    // Position of the first morsel sent by the reader.
    let first_morsel_position = if pre_slice.is_some() {
        // Pre-slice was pushed to reader.
        let Some(PhysicalSlice {
            slice: _,
            slice_start_position,
        }) = pre_slice_this_file
        else {
            panic!("{pre_slice_this_file:?}")
        };

        slice_start_position
    } else {
        RowCounter::default()
    };

    if verbose {
        eprintln!(
            "[ReaderStarter]: scan_source_idx: {scan_source_idx}: \
            projection_to_reader: {:?}, \
            projection_to_post: {:?}, \
            pre_slice_to_reader: {:?}, \
            external_filter_mask: {}",
            &projection_to_reader,
            &projection_to_post,
            pre_slice,
            ExternalFilterMask::log_display(external_filter_mask.as_ref()),
        )
    }

    let file_schema_rx = if forbid_extra_columns.is_some() {
        // Upstream should not have any reason to attach this.
        assert!(callbacks.file_schema_tx.is_none());
        let (tx, rx) = connector::connector();
        callbacks.file_schema_tx = Some(tx);
        Some(rx)
    } else {
        None
    };

    // Should not have both of these set, as the `n_rows_in_file` will cause the `row_position_on_end`
    // callback to be unnecessarily blocked in CSV and NDJSON.
    debug_assert!(
        !(callbacks.row_position_on_end_tx.is_some() && callbacks.n_rows_in_file_tx.is_some()),
    );

    if let Some(predicate) = predicate.as_mut() {
        assert!(matches!(projection_to_post, Projection::Plain(_)));

        let reader_file_schema = reader.file_schema().await?;

        // If we are sending a filter into the reader, fully initialize and resolve the projection
        // here (i.e. column renaming / casting).
        projection_to_reader = match projection_to_reader {
            Projection::Plain(projected_schema) => {
                assert!(file_iceberg_schema.is_none());
                assert!(matches!(
                    file_projection_builder,
                    ProjectionBuilder::Plain(_)
                ));
                assert!(matches!(projection_to_post, Projection::Plain(_)));

                ProjectionBuilder::new(projected_schema, None).build_projection(
                    Some(reader_file_schema.as_ref()),
                    None,
                    cast_columns_policy.clone(),
                )?
            },
            Projection::Mapped { .. } => projection_to_reader,
        };

        let mut external_predicate_cols = Vec::with_capacity(
            hive_parts.as_ref().map_or(0, |x| x.df().width())
                + extra_ops_post.include_file_paths.is_some() as usize
                + projection_to_reader.num_missing_columns().unwrap(),
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

        for (missing_col_name, dtype) in
            file_projection.iter_missing_columns(Some(&reader_file_schema))?
        {
            match &missing_columns_policy {
                MissingColumnsPolicy::Insert => external_predicate_cols
                    .push((missing_col_name.clone(), Scalar::null(dtype.clone()))),
                MissingColumnsPolicy::Raise => return Err(missing_column_err(missing_col_name)),
            }
        }

        predicate.set_external_constant_columns(external_predicate_cols);
    }

    let begin_read_args = BeginReadArgs {
        projection: projection_to_reader,
        row_index,
        pre_slice,
        predicate,
        cast_columns_policy: cast_columns_policy.clone(),
        num_pipelines,
        callbacks,
    };

    let (mut reader_output_port, reader_handle) = reader.begin_read(begin_read_args)?;

    let reader_handle = AbortOnDropHandle::new(reader_handle);

    if let Some(forbid_extra_columns) = forbid_extra_columns {
        if let Ok(this_file_schema) = file_schema_rx.unwrap().recv().await {
            forbid_extra_columns
                .check_file_schema(&this_file_schema, file_iceberg_schema.as_ref())?;
        } else {
            drop(reader_output_port);
            return Err(reader_handle.await.unwrap_err());
        }
    }

    let first_morsel = reader_output_port.recv().await.ok();

    let ops_applier = if let Some(first_morsel) = &first_morsel {
        let final_output_schema = final_output_schema.clone();
        let extra_ops = extra_ops_post;

        if verbose {
            eprintln!(
                "start_reader_impl: \
                scan_source_idx: {scan_source_idx}, \
                first_morsel_position: {first_morsel_position:?}"
            )
        }

        ApplyExtraOps::Uninitialized {
            final_output_schema,
            projection: projection_to_post,
            cast_columns_policy,
            missing_columns_policy,
            extra_ops,
            scan_source: scan_source.clone(),
            scan_source_idx,
            hive_parts,
            external_filter_mask,
        }
        .initialize(first_morsel.df().schema())?
    } else {
        ApplyExtraOps::Noop
    };

    // Note: We assume that if we have an Initialized ops_applier, then the first_morsel is Some(_).

    if verbose {
        eprintln!(
            "start_reader_impl: \
            scan_source_idx: {scan_source_idx}, \
            ApplyExtraOps::{}, \
            first_morsel_position: {first_morsel_position:?}",
            ops_applier.variant_name(),
        );
    }

    let (bridge_recv_port, post_apply_pipeline_handle) = match ops_applier {
        ApplyExtraOps::Initialized { .. } => {
            let ops_applier = Arc::new(ops_applier);
            let first_morsel = first_morsel.unwrap();

            let (rx, handle) = PostApplyPipeline {
                reader_output_port,
                ops_applier,
                first_morsel,
                first_morsel_position,
                num_pipelines,
            }
            .run();

            (BridgeRecvPort::Linearized { rx }, Some(handle))
        },

        ApplyExtraOps::Noop => (
            BridgeRecvPort::Direct {
                rx: reader_output_port,
                first_morsel,
            },
            None,
        ),

        ApplyExtraOps::Uninitialized { .. } => unreachable!(),
    };

    let state = StartedReaderState {
        bridge_recv_port,
        post_apply_pipeline_handle,
        reader_handle,
    };

    Ok(state)
}

/// State for a reader that has been started.
struct StartedReaderState {
    bridge_recv_port: BridgeRecvPort,
    post_apply_pipeline_handle: Option<AbortOnDropHandle<PolarsResult<()>>>,
    reader_handle: AbortOnDropHandle<PolarsResult<()>>,
}

struct AttachReaderToBridge {
    started_reader_rx: tokio::sync::mpsc::Receiver<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
    verbose: bool,
}

impl AttachReaderToBridge {
    async fn run(self) -> PolarsResult<()> {
        let AttachReaderToBridge {
            mut started_reader_rx,
            mut bridge_recv_port_tx,
            verbose,
        } = self;

        let mut n_readers_received: usize = 0;

        while let Some((init_task_handle, wait_token)) = started_reader_rx.recv().await {
            n_readers_received = n_readers_received.saturating_add(1);

            if verbose {
                eprintln!(
                    "[AttachReaderToBridge]: received reader (n_readers_received: {n_readers_received})",
                );
            }

            let StartedReaderState {
                bridge_recv_port,
                post_apply_pipeline_handle,
                reader_handle,
            } = init_task_handle.await?;

            if bridge_recv_port_tx.send(bridge_recv_port).await.is_err() {
                break;
            }

            drop(wait_token);
            reader_handle.await?;

            if let Some(handle) = post_apply_pipeline_handle {
                handle.await?;
            }
        }

        Ok(())
    }
}

/// Encapsulates logic for determining which operations to push into the underlying reader.
struct ReaderOperationPushdown<'a> {
    file_projection: Projection,
    reader_capabilities: ReaderCapabilities,
    external_filter_mask: Option<ExternalFilterMask>,
    /// Operations will be `take()`en out when pushed.
    extra_ops_post: &'a mut ExtraOperations,
}

impl ReaderOperationPushdown<'_> {
    fn push_operations(
        self,
    ) -> (
        Projection,
        Projection,
        Option<RowIndex>,
        Option<Slice>,
        Option<ScanIOPredicate>,
        Option<ExternalFilterMask>,
    ) {
        let Self {
            file_projection,
            reader_capabilities,
            external_filter_mask,
            extra_ops_post,
        } = self;

        use ReaderCapabilities as RC;

        let unsupported_external_filter_mask = external_filter_mask.is_some()
            && !reader_capabilities.contains(RC::EXTERNAL_FILTER_MASK);

        let unsupported_resolved_mapped_projection = match &file_projection {
            Projection::Plain(_) => false,
            Projection::Mapped { .. } => {
                !reader_capabilities.contains(RC::MAPPED_COLUMN_PROJECTION)
            },
        };

        let (projection_to_reader, projection_to_post) = if unsupported_resolved_mapped_projection {
            (file_projection.get_plain_pre_projection(), file_projection)
        } else {
            let projection_to_post = Projection::Plain(file_projection.projected_schema().clone());
            (file_projection, projection_to_post)
        };

        // Notes
        // * If there is both a slice and deletions, DO NOT push deletions to the reader without
        //   pushing the slice.

        // If `unsupported_mapped_projection`, the file may contain a column sharing the name of
        // the row index column, but gets renamed by the column mapping.
        let row_index = if reader_capabilities.contains(RC::ROW_INDEX)
            && !(unsupported_resolved_mapped_projection || unsupported_external_filter_mask)
        {
            extra_ops_post.row_index.take()
        } else {
            None
        };

        let pre_slice = match &extra_ops_post.pre_slice {
            Some(Slice::Positive { .. }) if reader_capabilities.contains(RC::PRE_SLICE) => {
                extra_ops_post.pre_slice.take()
            },

            Some(Slice::Negative { .. })
                if reader_capabilities.contains(RC::NEGATIVE_PRE_SLICE) =>
            {
                extra_ops_post.pre_slice.take()
            },

            _ => None,
        };

        let push_predicate = !(!reader_capabilities.contains(RC::MAPPED_COLUMN_PROJECTION)
            || unsupported_external_filter_mask
            || extra_ops_post.predicate.is_none()
            || (extra_ops_post.row_index.is_some() || extra_ops_post.pre_slice.is_some())
            || !reader_capabilities.contains(RC::PARTIAL_FILTER));

        let mut predicate: Option<ScanIOPredicate> = None;

        if push_predicate {
            predicate = if reader_capabilities.contains(RC::FULL_FILTER) {
                // If the reader can fully handle the predicate itself, let it do it itself.
                extra_ops_post.predicate.take()
            } else {
                // Otherwise, we want to pass it and filter again afterwards.
                extra_ops_post.predicate.clone()
            }
        }

        (
            projection_to_reader,
            projection_to_post,
            row_index,
            pre_slice,
            predicate,
            external_filter_mask,
        )
    }
}

/// Represents a [`Slice`] that has been potentially adjusted to account for deleted rows.
#[derive(Debug)]
struct PhysicalSlice {
    slice: Slice,
    /// Counter that records the number of physical and deleted rows that make up the slice offset,
    /// and `slice_start_position.num_rows() == slice.offset`
    slice_start_position: RowCounter,
}

impl PhysicalSlice {
    /// # Panics
    /// Panics if `slice` is [`Slice::Negative`]
    fn new(slice: Slice, external_filter_mask: Option<&ExternalFilterMask>) -> Self {
        if let Some(external_filter_mask) = external_filter_mask {
            let requested_offset = slice.positive_offset();

            let physical_slice = external_filter_mask.calc_physical_slice(slice);

            let physical_offset = physical_slice.positive_offset();
            let deleted_in_offset = physical_offset.checked_sub(requested_offset).unwrap();

            let slice_start_position = RowCounter::new(physical_offset, deleted_in_offset);

            PhysicalSlice {
                slice: physical_slice,
                slice_start_position,
            }
        } else {
            let slice_start_position = RowCounter::new(slice.positive_offset(), 0);

            PhysicalSlice {
                slice,
                slice_start_position,
            }
        }
    }
}
