use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::{HiveOptions, RowIndex};
use polars_plan::dsl::{
    CastColumnsPolicy, DslPlan, ExtraColumnsPolicy, FileScanDsl, MissingColumnsPolicy, ScanSources,
};
use polars_plan::prelude::{NDJsonReadOptions, UnifiedScanArgs};
use polars_utils::plpath::PlPath;
use polars_utils::slice_enum::Slice;

use crate::prelude::LazyFrame;
use crate::scan::file_list_reader::LazyFileListReader;

#[derive(Clone)]
pub struct LazyJsonLineReader {
    pub(crate) sources: ScanSources,
    pub(crate) batch_size: Option<NonZeroUsize>,
    pub(crate) low_memory: bool,
    pub(crate) rechunk: bool,
    pub(crate) schema: Option<SchemaRef>,
    pub(crate) schema_overwrite: Option<SchemaRef>,
    pub(crate) row_index: Option<RowIndex>,
    pub(crate) infer_schema_length: Option<NonZeroUsize>,
    pub(crate) n_rows: Option<usize>,
    pub(crate) ignore_errors: bool,
    pub(crate) include_file_paths: Option<PlSmallStr>,
    pub(crate) cloud_options: Option<CloudOptions>,
}

impl LazyJsonLineReader {
    pub fn new_paths(paths: Arc<[PlPath]>) -> Self {
        Self::new_with_sources(ScanSources::Paths(paths))
    }

    pub fn new_with_sources(sources: ScanSources) -> Self {
        LazyJsonLineReader {
            sources,
            batch_size: None,
            low_memory: false,
            rechunk: false,
            schema: None,
            schema_overwrite: None,
            row_index: None,
            infer_schema_length: NonZeroUsize::new(100),
            ignore_errors: false,
            n_rows: None,
            include_file_paths: None,
            cloud_options: None,
        }
    }

    pub fn new(path: PlPath) -> Self {
        Self::new_with_sources(ScanSources::Paths([path].into()))
    }

    /// Add a row index column.
    #[must_use]
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Set values as `Null` if parsing fails because of schema mismatches.
    #[must_use]
    pub fn with_ignore_errors(mut self, ignore_errors: bool) -> Self {
        self.ignore_errors = ignore_errors;
        self
    }
    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    #[must_use]
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }
    /// Set the number of rows to use when inferring the json schema.
    /// the default is 100 rows.
    /// Ignored when the schema is specified explicitly using [`Self::with_schema`].
    /// Setting to `None` will do a full table scan, very slow.
    #[must_use]
    pub fn with_infer_schema_length(mut self, num_rows: Option<NonZeroUsize>) -> Self {
        self.infer_schema_length = num_rows;
        self
    }
    /// Set the JSON file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.schema = schema;
        self
    }

    /// Set the JSON file's schema
    #[must_use]
    pub fn with_schema_overwrite(mut self, schema_overwrite: Option<SchemaRef>) -> Self {
        self.schema_overwrite = schema_overwrite;
        self
    }

    /// Reduce memory usage at the expense of performance
    #[must_use]
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: Option<NonZeroUsize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_cloud_options(mut self, cloud_options: Option<CloudOptions>) -> Self {
        self.cloud_options = cloud_options;
        self
    }

    pub fn with_include_file_paths(mut self, include_file_paths: Option<PlSmallStr>) -> Self {
        self.include_file_paths = include_file_paths;
        self
    }
}

impl LazyFileListReader for LazyJsonLineReader {
    fn finish(self) -> PolarsResult<LazyFrame> {
        let unified_scan_args = UnifiedScanArgs {
            schema: None,
            cloud_options: self.cloud_options,
            hive_options: HiveOptions::new_disabled(),
            rechunk: self.rechunk,
            cache: false,
            glob: true,
            projection: None,
            row_index: self.row_index,
            pre_slice: self.n_rows.map(|len| Slice::Positive { offset: 0, len }),
            cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
            missing_columns_policy: MissingColumnsPolicy::Raise,
            extra_columns_policy: ExtraColumnsPolicy::Raise,
            include_file_paths: self.include_file_paths,
            column_mapping: None,
            deletion_files: None,
        };

        let options = NDJsonReadOptions {
            n_threads: None,
            infer_schema_length: self.infer_schema_length,
            chunk_size: NonZeroUsize::new(1 << 18).unwrap(),
            low_memory: self.low_memory,
            ignore_errors: self.ignore_errors,
            schema: self.schema,
            schema_overwrite: self.schema_overwrite,
        };

        let scan_type = Box::new(FileScanDsl::NDJson { options });

        Ok(LazyFrame::from(DslPlan::Scan {
            sources: self.sources,
            unified_scan_args: Box::new(unified_scan_args),
            scan_type,
            cached_ir: Default::default(),
        }))
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        unreachable!();
    }

    fn sources(&self) -> &ScanSources {
        &self.sources
    }

    fn with_sources(mut self, sources: ScanSources) -> Self {
        self.sources = sources;
        self
    }

    fn with_n_rows(mut self, n_rows: impl Into<Option<usize>>) -> Self {
        self.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.row_index = row_index.into();
        self
    }

    fn rechunk(&self) -> bool {
        self.rechunk
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.rechunk = toggle;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize> {
        self.n_rows
    }

    /// Add a row index column.
    fn row_index(&self) -> Option<&RowIndex> {
        self.row_index.as_ref()
    }

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.cloud_options.as_ref()
    }
}
