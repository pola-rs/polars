use std::collections::HashMap;
use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::file_format::ObjectInfo;
use polars_io::input::file_format::parquet::ParquetFormat;
use polars_io::input::file_format::FileFormat;
use polars_io::input::file_listing::ObjectListingUrl;
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
        }
    }
}

#[derive(Clone)]
struct LazyParquetReader {
    path: PathBuf,
    args: ScanArgsParquet,
    path_str: String,
}

impl LazyParquetReader {
    fn new(path: PathBuf, args: ScanArgsParquet) -> Self {
        Self {
            path,
            args,
            path_str: String::from(""),
        }
    }

    fn new2(path: String, args: ScanArgsParquet) -> Self {
        Self {
            path: PathBuf::from(path.clone()),
            args,
            path_str: path,
        }
    }
}

impl LazyFileListReader for LazyParquetReader {
    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
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
        )?
        .build()
        .into();

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }

        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    fn glob_object_infos(self) -> PolarsResult<Vec<ObjectInfo>> {
        let path_str = self.path_str.as_str();
        let url = ObjectListingUrl::parse(path_str)?;

        // todo! get this from cloud options
        let cloud_opts = HashMap::new();
        ParquetFormat::create().glob_object_info(url, cloud_opts, true, false)
    }

    fn object_to_lazy(self, file_info: ObjectInfo) -> PolarsResult<LazyFrame> {
        let (path, schema, row_estimation) = file_info;

        let file_info = FileInfo {
            schema: Arc::new(schema),
            row_estimation,
        };

        let row_count = self.row_count().map(|x| x.to_owned());

        // todo! check if this is still needed
        // if let Some(rc) = self.row_count() {
        //     let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
        // }

        let options = FileScanOptions {
            with_columns: None,
            cache: self.args.cache,
            n_rows: self.args.n_rows,
            rechunk: false,
            row_count: row_count.clone(),
            file_counter: Default::default(),
        };

        let lpb: LogicalPlanBuilder = LogicalPlan::Scan {
            path: PathBuf::from(path),
            file_info,
            file_options: options,
            predicate: None,
            scan_type: FileScan::Parquet {
                options: ParquetOptions {
                    parallel: self.args.parallel,
                    low_memory: self.args.low_memory,
                    use_statistics: self.args.use_statistics,
                },
                cloud_options: self.args.cloud_options,
            },
        }
        .into();

        let mut lf: LazyFrame = lpb.build().into();

        // todo! should this be done post concat?
        // it is a bit hacky, but this row_count function updates the schema
        if let Some(rc) = &row_count {
            lf = lf.with_row_count(&rc.name, Some(rc.offset))
        }

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

    fn n_rows(&self) -> Option<usize> {
        self.args.n_rows
    }

    fn row_count(&self) -> Option<&RowCount> {
        self.args.row_count.as_ref()
    }

    fn cloud_options(&self) -> Option<&CloudOptions> {
        self.args.cloud_options.as_ref()
    }
}

impl LazyFrame {
    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new(path.as_ref().to_owned(), args).finish()
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet2(path: &str, args: ScanArgsParquet) -> PolarsResult<Self> {
        LazyParquetReader::new2(path.to_string(), args).finish2()
    }
}
