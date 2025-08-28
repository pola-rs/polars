use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::deletion::DeletionFilesList;
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy, ScanSources};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::relaxed_cell::RelaxedCell;
use polars_utils::slice_enum::Slice;
use reader_interface::builder::FileReaderBuilder;
use reader_interface::capabilities::ReaderCapabilities;

use crate::nodes::io_sources::multi_scan::components::forbid_extra_columns::ForbidExtraColumns;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface;

// Some parts are called MultiScan for now to avoid conflict with existing MultiScan.

pub struct MultiScanConfig {
    pub sources: ScanSources,
    pub file_reader_builder: Arc<dyn FileReaderBuilder>,
    pub cloud_options: Option<Arc<CloudOptions>>,

    /// Final output schema of MultiScan node. Includes all e.g. row index / missing columns / file paths / hive etc.
    pub final_output_schema: SchemaRef,
    /// Columns to be projected from the file.
    pub file_projection_builder: ProjectionBuilder,

    pub row_index: Option<RowIndex>,
    pub pre_slice: Option<Slice>,
    pub predicate: Option<ScanIOPredicate>,

    pub hive_parts: Option<Arc<HivePartitionsDf>>,
    pub include_file_paths: Option<PlSmallStr>,
    pub missing_columns_policy: MissingColumnsPolicy,
    pub cast_columns_policy: CastColumnsPolicy,
    pub forbid_extra_columns: Option<ForbidExtraColumns>,
    pub deletion_files: Option<DeletionFilesList>,

    pub num_pipelines: RelaxedCell<usize>,
    /// Number of readers to initialize concurrently. e.g. Parquet will want to fetch metadata in this
    /// step.
    pub n_readers_pre_init: RelaxedCell<usize>,
    pub max_concurrent_scans: RelaxedCell<usize>,

    pub verbose: bool,
}

impl MultiScanConfig {
    pub fn num_pipelines(&self) -> usize {
        self.num_pipelines.load()
    }

    pub fn n_readers_pre_init(&self) -> usize {
        self.n_readers_pre_init.load()
    }

    pub fn max_concurrent_scans(&self) -> usize {
        self.max_concurrent_scans.load()
    }

    pub fn reader_capabilities(&self) -> ReaderCapabilities {
        if std::env::var("POLARS_FORCE_EMPTY_READER_CAPABILITIES").as_deref() == Ok("1") {
            self.file_reader_builder.reader_capabilities()
                & ReaderCapabilities::NEEDS_FILE_CACHE_INIT
        } else {
            self.file_reader_builder.reader_capabilities()
        }
    }
}
