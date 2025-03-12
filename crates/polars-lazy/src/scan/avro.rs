use std::path::Path;

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::{HiveOptions, RowIndex};
use polars_plan::dsl::{DslPlan, FileScan, ScanSources};
use polars_plan::prelude::FileScanOptions;

use crate::prelude::LazyFrame;
use crate::scan::file_list_reader::LazyFileListReader;

#[derive(Clone)]
pub struct LazyAvroReader {
    pub(crate) sources: ScanSources,
    pub(crate) rechunk: bool,
    pub(crate) row_index: Option<RowIndex>,
    pub(crate) n_rows: Option<usize>,
    pub(crate) include_file_paths: Option<PlSmallStr>,
    pub(crate) cloud_options: Option<CloudOptions>,
}

impl LazyAvroReader {
    #[must_use]
    pub fn new_with_sources(sources: ScanSources) -> Self {
        Self {
            sources,
            rechunk: false,
            row_index: None,
            n_rows: None,
            include_file_paths: None,
            cloud_options: None,
        }
    }

    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::new_with_sources(ScanSources::Paths([path.as_ref().to_path_buf()].into()))
    }

    /// Add a row index column.
    #[must_use]
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    #[must_use]
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    #[must_use]
    pub fn with_cloud_options(mut self, cloud_options: Option<CloudOptions>) -> Self {
        self.cloud_options = cloud_options;
        self
    }

    #[must_use]
    pub fn with_include_file_paths(mut self, include_file_paths: Option<PlSmallStr>) -> Self {
        self.include_file_paths = include_file_paths;
        self
    }
}

impl LazyFileListReader for LazyAvroReader {
    fn finish(self) -> PolarsResult<LazyFrame> {
        let file_options = Box::new(FileScanOptions {
            pre_slice: self.n_rows.map(|x| (0, x)),
            with_columns: None,
            cache: false,
            row_index: self.row_index,
            rechunk: self.rechunk,
            file_counter: 0,
            hive_options: HiveOptions {
                enabled: Some(false),
                hive_start_idx: 0,
                schema: None,
                try_parse_dates: true,
            },
            glob: true,
            include_file_paths: self.include_file_paths,
            allow_missing_columns: false,
        });

        let scan_type = Box::new(FileScan::Avro {
            cloud_options: self.cloud_options,
        });

        Ok(LazyFrame::from(DslPlan::Scan {
            sources: self.sources,
            file_info: None,
            file_options,
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
