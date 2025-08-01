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
