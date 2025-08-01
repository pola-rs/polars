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
use crate::nodes::io_sources::multi_file_reader::components::apply_extra_ops::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::components::row_deletions::{
    DeletionFilesProvider, ExternalFilterMask, RowDeletionsInit,
};
use crate::nodes::io_sources::multi_file_reader::components::{
    ExtraOperations, ForbidExtraColumns, missing_column_err,
};
use crate::nodes::io_sources::multi_file_reader::functions::initialize_multi_scan_pipeline;
use crate::nodes::io_sources::multi_file_reader::functions::resolve_projections::ProjectionBuilder;
use crate::nodes::io_sources::multi_file_reader::functions::resolve_slice::{
    ResolvedSliceInfo, resolve_to_positive_slice,
};
use crate::nodes::io_sources::multi_file_reader::post_apply_pipeline::PostApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};
use crate::nodes::io_sources::multi_file_reader::row_counter::RowCounter;

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
