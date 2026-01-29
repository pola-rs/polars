use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::read::ParallelStrategy;
use polars_io::prelude::ParquetOptions;
use polars_io::{HiveOptions, RowIndex};
use polars_utils::pl_path::PlRefPath;
use polars_utils::slice_enum::Slice;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub parallel: ParallelStrategy,
    pub row_index: Option<RowIndex>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,
    pub use_statistics: bool,
    pub schema: Option<SchemaRef>,
    pub low_memory: bool,
    pub rechunk: bool,
    pub cache: bool,
    /// Expand path given via globbing rules.
    pub glob: bool,
    pub include_file_paths: Option<PlSmallStr>,
    pub allow_missing_columns: bool,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            parallel: Default::default(),
            row_index: None,
            cloud_options: None,
            hive_options: Default::default(),
            use_statistics: true,
            schema: None,
            rechunk: false,
            low_memory: false,
            cache: true,
            glob: true,
            include_file_paths: None,
            allow_missing_columns: false,
        }
    }
}

impl ScanArgsParquet {
    /// Setter for the `n_rows` field.
    #[inline(always)]
    pub fn n_rows(mut self, n_rows: Option<usize>) -> Self {
        self.n_rows = n_rows;
        self
    }
    /// Setter for the `cache` field.
    #[inline(always)]
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }
    /// Setter for the `parallel` field.
    #[inline(always)]
    pub fn parallel(mut self, parallel: ParallelStrategy) -> Self {
        self.parallel = parallel;
        self
    }
    /// Setter for the `rechunk` field.
    #[inline(always)]
    pub fn rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }
    /// Setter for the `row_index` field.
    #[inline(always)]
    pub fn row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.row_index = row_index;
        self
    }
    /// Setter for the `low_memory` field.
    #[inline(always)]
    pub fn low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }
    /// Setter for the `cloud_options` field.
    #[inline(always)]
    pub fn cloud_options(mut self, cloud_options: Option<CloudOptions>) -> Self {
        self.cloud_options = cloud_options;
        self
    }
    /// Setter for the `use_statistics` field.
    #[inline(always)]
    pub fn use_statistics(mut self, use_statistics: bool) -> Self {
        self.use_statistics = use_statistics;
        self
    }
    /// Setter for the `hive_options` field.
    #[inline(always)]
    pub fn hive_options(mut self, hive_options: HiveOptions) -> Self {
        self.hive_options = hive_options;
        self
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    args: ScanArgsParquet,
    sources: ScanSources,
}

impl LazyParquetReader {
    fn new(args: ScanArgsParquet) -> Self {
        Self {
            args,
            sources: ScanSources::default(),
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    /// Get the final [LazyFrame].
    fn finish(self) -> PolarsResult<LazyFrame> {
        let parquet_options = ParquetOptions {
            schema: self.args.schema,
            parallel: self.args.parallel,
            low_memory: self.args.low_memory,
            use_statistics: self.args.use_statistics,
        };

        let unified_scan_args = UnifiedScanArgs {
            schema: None,
            cloud_options: self.args.cloud_options,
            hive_options: self.args.hive_options,
            rechunk: self.args.rechunk,
            cache: self.args.cache,
            glob: self.args.glob,
            hidden_file_prefix: None,
            projection: None,
            column_mapping: None,
            default_values: None,
            // Note: We call `with_row_index()` on the LazyFrame below
            row_index: None,
            pre_slice: self
                .args
                .n_rows
                .map(|len| Slice::Positive { offset: 0, len }),
            cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
            missing_columns_policy: if self.args.allow_missing_columns {
                MissingColumnsPolicy::Insert
            } else {
                MissingColumnsPolicy::Raise
            },
            extra_columns_policy: ExtraColumnsPolicy::Raise,
            include_file_paths: self.args.include_file_paths,
            deletion_files: None,
            table_statistics: None,
            row_count: None,
        };

        let mut lf: LazyFrame =
            DslBuilder::scan_parquet(self.sources, parquet_options, unified_scan_args)?
                .build()
                .into();

        // It's a bit hacky, but this row_index function updates the schema.
        if let Some(row_index) = self.args.row_index {
            lf = lf.with_row_index(row_index.name, Some(row_index.offset))
        }

        Ok(lf)
    }

    fn glob(&self) -> bool {
        self.args.glob
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
        self.args.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.args.row_index = row_index.into();
        self
    }

    fn rechunk(&self) -> bool {
        self.args.rechunk
    }

    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.args.rechunk = toggle;
        self
    }

    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.args.cloud_options.as_ref()
    }

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_index(&self) -> Option<&RowIndex> {
        self.args.row_index.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: PlRefPath, args: ScanArgsParquet) -> PolarsResult<Self> {
        Self::scan_parquet_sources(ScanSources::Paths(Buffer::from_iter([path])), args)
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_sources(sources: ScanSources, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(args).with_sources(sources).finish()
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet_files(
        paths: Buffer<PlRefPath>,
        args: ScanArgsParquet,
    ) -> PolarsResult<Self> {
        Self::scan_parquet_sources(ScanSources::Paths(paths), args)
    }
}
