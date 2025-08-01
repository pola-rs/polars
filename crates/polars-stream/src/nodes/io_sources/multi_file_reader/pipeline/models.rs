//! Structures for holding data.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use arrow::bitmap::Bitmap;
use components::row_counter::RowCounter;
use components::row_deletions::ExternalFilterMask;
use futures::StreamExt;
use futures::stream::BoxStream;
use polars_core::prelude::{AnyValue, DataType, PlHashMap, PlHashMap};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_core::schema::iceberg::IcebergSchema;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy, ScanSource};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, AbortOnDropHandle, JoinHandle, TaskPriority};
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken, WaitToken};
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_file_reader::components;
use crate::nodes::io_sources::multi_file_reader::components::apply_extra_ops::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::components::bridge::BridgeState;
use crate::nodes::io_sources::multi_file_reader::components::physical_slice::PhysicalSlice;
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
    BeginReadArgs, FileReader, FileReader, FileReaderCallbacks, Projection,
};
use crate::nodes::io_sources::multi_file_reader::row_counter::RowCounter;

pub struct InitializedPipelineState {
    pub task_handle: AbortOnDropHandle<PolarsResult<()>>,
    pub phase_channel_tx: connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
    pub bridge_state: Arc<Mutex<BridgeState>>,
}

/// Anything aside from reading columns from the file. E.g. row_index, slice, predicate etc.
///
/// Note that hive partition columns are tracked separately.
#[derive(Debug, Default, Clone)]
pub(super) struct ExtraOperations {
    // Note: These fields are ordered according to when they (should be) applied.
    pub(super) row_index: Option<RowIndex>,
    /// Index of the row index column in the final output.
    pub(super) row_index_col_idx: usize,
    pub(super) pre_slice: Option<Slice>,
    pub(super) include_file_paths: Option<PlSmallStr>,
    /// Index of the file path column in the final output.
    pub(super) file_path_col_idx: usize,
    pub(super) predicate: Option<ScanIOPredicate>,
}

impl ExtraOperations {
    pub(super) fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}

pub(super) struct ResolvedSliceInfo {
    /// In the negative slice case this can be a non-zero starting position.
    pub(super) scan_source_idx: usize,
    /// In the negative slice case this will hold a row index with the offset adjusted.
    pub(super) row_index: Option<RowIndex>,
    /// Resolved positive slice.
    pub(super) pre_slice: Option<Slice>,
    /// If we resolved a negative slice we keep the initialized readers here (with a limit). For
    /// Parquet this can save a duplicate metadata fetch/decode.
    ///
    /// This will be in-order - i.e. `pop_front()` corresponds to the next reader.
    ///
    /// `Option(scan_source_idx, Deque(file_reader, n_rows))`
    #[expect(clippy::type_complexity)]
    pub(super) initialized_readers: Option<(usize, VecDeque<(Box<dyn FileReader>, RowCounter)>)>,
    pub(super) row_deletions: PlHashMap<usize, ExternalFilterMask>,
}

/// Constant over the file list.
#[derive(Clone)]
pub(super) struct StartReaderArgsConstant {
    pub(super) hive_parts: Option<Arc<HivePartitionsDf>>,
    pub(super) final_output_schema: SchemaRef,
    pub(super) reader_capabilities: ReaderCapabilities,
    pub(super) file_projection_builder: ProjectionBuilder,
    pub(super) cast_columns_policy: CastColumnsPolicy,
    pub(super) missing_columns_policy: MissingColumnsPolicy,
    pub(super) forbid_extra_columns: Option<ForbidExtraColumns>,
    pub(super) num_pipelines: usize,
    pub(super) verbose: bool,
}

pub(super) struct StartReaderArgsPerFile {
    pub(super) scan_source: ScanSource,
    pub(super) scan_source_idx: usize,
    pub(super) reader: Box<dyn FileReader>,
    pub(super) pre_slice_this_file: Option<PhysicalSlice>,
    pub(super) extra_ops_this_file: ExtraOperations,
    pub(super) callbacks: FileReaderCallbacks,
    pub(super) external_filter_mask: Option<ExternalFilterMask>,
}

/// State for a reader that has been started.
pub(super) struct StartedReaderState {
    pub(super) bridge_recv_port: BridgeRecvPort,
    pub(super) post_apply_pipeline_handle: Option<AbortOnDropHandle<PolarsResult<()>>>,
    pub(super) reader_handle: AbortOnDropHandle<PolarsResult<()>>,
}
