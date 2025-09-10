use std::sync::Arc;

use arrow::bitmap::Bitmap;
use components::bridge::BridgeRecvPort;
use components::row_deletions::{ExternalFilterMask, RowDeletionsInit};
use futures::StreamExt;
use futures::stream::BoxStream;
use polars_core::config::verbose_print_sensitive;
use polars_core::prelude::{AnyValue, DataType};
use polars_core::scalar::Scalar;
use polars_core::schema::iceberg::IcebergSchema;
use polars_error::PolarsResult;
use polars_plan::dsl::{MissingColumnsPolicy, ScanSource};
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::nodes::io_sources::multi_scan::components;
use crate::nodes::io_sources::multi_scan::components::apply_extra_ops::ApplyExtraOps;
use crate::nodes::io_sources::multi_scan::components::errors::missing_column_err;
use crate::nodes::io_sources::multi_scan::components::physical_slice::PhysicalSlice;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::components::reader_operation_pushdown::ReaderOperationPushdown;
use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::pipeline::models::{
    ExtraOperations, StartReaderArgsConstant, StartReaderArgsPerFile, StartedReaderState,
};
use crate::nodes::io_sources::multi_scan::pipeline::tasks::post_apply_extra_ops::PostApplyExtraOps;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};

/// Starts readers, potentially multiple at the same time if it can.
pub struct ReaderStarter {
    pub reader_capabilities: ReaderCapabilities,
    pub readers_init_iter: BoxStream<'static, PolarsResult<InitializedReaderState>>,
    pub n_sources: usize,
    pub started_reader_tx: tokio::sync::mpsc::Sender<(
        AbortOnDropHandle<PolarsResult<StartedReaderState>>,
        WaitToken,
    )>,
    pub max_concurrent_scans: usize,
    pub skip_files_mask: Option<Bitmap>,
    pub extra_ops: ExtraOperations,
    pub constant_args: StartReaderArgsConstant,
    pub verbose: bool,
}

pub struct InitializedReaderState {
    pub scan_source_idx: usize,
    pub scan_source: ScanSource,
    pub reader: Box<dyn FileReader>,
    pub n_rows_in_file: Option<RowCounter>,
    pub row_deletions: Option<RowDeletionsInit>,
}

impl ReaderStarter {
    pub async fn run(self) -> PolarsResult<()> {
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

            let should_update_row_position =
                extra_ops.has_row_index_or_slice() && n_sources - scan_source_idx > 1;

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
                if should_update_row_position {
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

            let (row_position_on_end_tx, row_position_on_end_rx) =
                if should_update_row_position && n_rows_in_file.is_none() {
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
        scan_source_idx,
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
            "[ReaderStarter]: \
            scan_source_idx: {scan_source_idx}: \
            pre_slice_to_reader: {:?}, \
            external_filter_mask: {}, \
            file_iceberg_schema: {:?}",
            pre_slice,
            ExternalFilterMask::log_display(external_filter_mask.as_ref()),
            &file_iceberg_schema,
        )
    }

    verbose_print_sensitive(|| {
        format!(
            "[ReaderStarter]: \
            projection_to_reader: {projection_to_reader:?}, \
            projection_to_post: {projection_to_post:?}"
        )
    });

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
        assert!(!projection_to_post.has_projection_transforms());

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

                ProjectionBuilder::new(projected_schema, None, None).build_projection(
                    Some(reader_file_schema.as_ref()),
                    None,
                    cast_columns_policy.clone(),
                    scan_source_idx,
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

        for (missing_col_name, dtype, default_value) in
            file_projection.iter_missing_columns(Some(&reader_file_schema))?
        {
            match &missing_columns_policy {
                MissingColumnsPolicy::Insert => {
                    if predicate.live_columns.contains(missing_col_name) {
                        external_predicate_cols.push((
                            missing_col_name.clone(),
                            default_value
                                .cloned()
                                .unwrap_or_else(|| Scalar::null(dtype.clone())),
                        ));

                        Arc::make_mut(&mut predicate.column_predicates).is_sumwise_complete = false;
                    }
                },
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

            let (rx, handle) = PostApplyExtraOps {
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
