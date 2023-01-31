use std::path::{Path, PathBuf};

use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
#[cfg(feature = "async")]
use polars_io::async_glob;
use polars_io::parquet::ParallelStrategy;
use polars_io::{is_cloud_url, RowCount};

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
        }
    }
}

impl LazyFrame {
    #[allow(clippy::too_many_arguments)]
    fn scan_parquet_impl(
        path: impl AsRef<Path>,
        n_rows: Option<usize>,
        cache: bool,
        parallel: ParallelStrategy,
        row_count: Option<RowCount>,
        rechunk: bool,
        low_memory: bool,
        cloud_options: Option<CloudOptions>,
    ) -> PolarsResult<Self> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(
            path.as_ref(),
            n_rows,
            cache,
            parallel,
            None,
            rechunk,
            low_memory,
            cloud_options,
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

    fn concat_impl(lfs: Vec<LazyFrame>, args: ScanArgsParquet) -> PolarsResult<LazyFrame> {
        concat_impl(&lfs, args.rechunk, true, true).map(|mut lf| {
            if let Some(n_rows) = args.n_rows {
                lf = lf.slice(0, n_rows as IdxSize)
            };
            if let Some(rc) = args.row_count {
                lf = lf.with_row_count(&rc.name, Some(rc.offset))
            };
            lf
        })
    }

    /// Create a LazyFrame directly from a parquet scan.
    #[deprecated(note = "please use `concat_lf` instead")]
    pub fn scan_parquet_files<P: AsRef<Path>>(
        paths: Vec<P>,
        args: ScanArgsParquet,
    ) -> PolarsResult<Self> {
        let lfs = paths
            .iter()
            .map(|p| {
                Self::scan_parquet_impl(
                    p,
                    args.n_rows,
                    args.cache,
                    args.parallel,
                    None,
                    args.rechunk,
                    args.low_memory,
                    args.cloud_options.clone(),
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        Self::concat_impl(lfs, args)
    }

    /// Create a LazyFrame directly from a parquet scan.
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();
        if path_str.contains('*') {
            let paths = if is_cloud_url(path) {
                #[cfg(feature = "async")]
                {
                    Box::new(
                        async_glob(&path_str, args.cloud_options.as_ref())?
                            .into_iter()
                            .map(|a| Ok(PathBuf::from(&a))),
                    )
                }
                #[cfg(not(feature = "async"))]
                panic!("Feature `async` must be enabled to use globbing patterns with cloud urls.")
            } else {
                Box::new(
                    glob::glob(&path_str).map_err(|_| {
                        PolarsError::ComputeError("invalid glob pattern given".into())
                    })?,
                ) as Box<dyn Iterator<Item = Result<PathBuf, _>>>
            };
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
                    Self::scan_parquet_impl(
                        path.clone(),
                        args.n_rows,
                        args.cache,
                        ParallelStrategy::None,
                        None,
                        false,
                        args.low_memory,
                        args.cloud_options.clone(),
                    )
                    .map_err(|e| {
                        PolarsError::ComputeError(
                            format!("While reading {} got {e:?}.", path.display()).into(),
                        )
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            if lfs.is_empty() {
                return PolarsResult::Err(PolarsError::ComputeError(
                    format!("Could not load any dataframes from {path_str}").into(),
                ));
            }
            Self::concat_impl(lfs, args)
        } else {
            Self::scan_parquet_impl(
                path,
                args.n_rows,
                args.cache,
                args.parallel,
                args.row_count,
                args.rechunk,
                args.low_memory,
                args.cloud_options,
            )
        }
    }
}
