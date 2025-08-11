//! Structures for holding data.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use components::row_counter::RowCounter;
use components::row_deletions::ExternalFilterMask;
use polars_core::prelude::PlHashMap;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy, ScanSource};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::async_executor::AbortOnDropHandle;
use crate::async_primitives::connector;
use crate::async_primitives::wait_group::WaitToken;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_scan::components;
use crate::nodes::io_sources::multi_scan::components::bridge::{BridgeRecvPort, BridgeState};
use crate::nodes::io_sources::multi_scan::components::forbid_extra_columns::ForbidExtraColumns;
use crate::nodes::io_sources::multi_scan::components::physical_slice::PhysicalSlice;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_scan::reader_interface::{FileReader, FileReaderCallbacks};

pub struct InitializedPipelineState {
    pub task_handle: AbortOnDropHandle<PolarsResult<()>>,
    pub phase_channel_tx: connector::Sender<(connector::Sender<Morsel>, WaitToken)>,
    pub bridge_state: Arc<Mutex<BridgeState>>,
}

/// Anything aside from reading columns from the file. E.g. row_index, slice, predicate etc.
///
/// Note that hive partition columns are tracked separately.
#[derive(Debug, Default, Clone)]
pub struct ExtraOperations {
    // Note: These fields are ordered according to when they (should be) applied.
    pub row_index: Option<RowIndex>,
    /// Index of the row index column in the final output.
    pub row_index_col_idx: usize,
    pub pre_slice: Option<Slice>,
    pub include_file_paths: Option<PlSmallStr>,
    /// Index of the file path column in the final output.
    pub file_path_col_idx: usize,
    pub predicate: Option<ScanIOPredicate>,
}

impl ExtraOperations {
    pub(super) fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}

pub struct ResolvedSliceInfo {
    /// In the negative slice case this can be a non-zero starting position.
    pub scan_source_idx: usize,
    /// In the negative slice case this will hold a row index with the offset adjusted.
    pub row_index: Option<RowIndex>,
    /// Resolved positive slice.
    pub pre_slice: Option<Slice>,
    /// If we resolved a negative slice we keep the initialized readers here (with a limit). For
    /// Parquet this can save a duplicate metadata fetch/decode.
    ///
    /// This will be in-order - i.e. `pop_front()` corresponds to the next reader.
    ///
    /// `Option(scan_source_idx, Deque(file_reader, n_rows))`
    #[expect(clippy::type_complexity)]
    pub initialized_readers: Option<(usize, VecDeque<(Box<dyn FileReader>, RowCounter)>)>,
    pub row_deletions: PlHashMap<usize, ExternalFilterMask>,
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
