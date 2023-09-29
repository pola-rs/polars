use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::parquet::ParallelStrategy;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: ParallelStrategy,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub low_memory: bool,
    pub cloud_options: Option<CloudOptions>,
    pub use_statistics: bool,
    pub hive_partitioning: bool,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            parallel: Default::default(),
            rechunk: true,
            row_count: None,
            low_memory: false,
            cloud_options: None,
            use_statistics: true,
            hive_partitioning: false,
        }
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    args: ScanArgsParquet,
    path: PathBuf,
    known_schema: Option<SchemaRef>,
}

impl LazyParquetReader {
    fn new(path: PathBuf, args: ScanArgsParquet) -> Self {
        Self {
            args,
            path,
            known_schema: None,
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    fn finish_no_glob(mut self) -> PolarsResult<LazyFrame> {
        let known_schema = self.known_schema();
        let row_count = self.args.row_count;
        let path = self.path;
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(
            path,
            self.args.n_rows,
            self.args.cache,
            self.args.parallel,
            None,
            self.args.rechunk,
            self.args.low_memory,
            self.args.cloud_options,
            self.args.use_statistics,
            self.args.hive_partitioning,
            known_schema,
        )?
        .build()
        .into();

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }
        self.known_schema = Some(lf.schema()?);

        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
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

    fn known_schema(&self) -> Option<SchemaRef> {
        self.known_schema.clone()
    }
    fn set_known_schema(&mut self, known_schema: SchemaRef) {
        self.known_schema = Some(known_schema);
    }

    fn row_count(&self) -> Option<&RowCount> {
        self.args.row_count.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(path.as_ref().to_owned(), args).finish()
    }
}
