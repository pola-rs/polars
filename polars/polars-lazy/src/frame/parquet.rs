use std::path::Path;

use polars_core::prelude::*;
use polars_io::parquet::ParallelStrategy;
use polars_io::RowCount;

use crate::dsl::functions::concat;
use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: ParallelStrategy,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub low_memory: bool,
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
        }
    }
}

impl LazyFrame {
    fn scan_parquet_impl(
        path: impl AsRef<Path>,
        n_rows: Option<usize>,
        cache: bool,
        parallel: ParallelStrategy,
        row_count: Option<RowCount>,
        rechunk: bool,
        low_memory: bool,
    ) -> PolarsResult<Self> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(
            path.as_ref(),
            n_rows,
            cache,
            parallel,
            None,
            rechunk,
            low_memory,
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
        concat(&lfs, args.rechunk, true).map(|mut lf| {
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
    #[cfg_attr(docsrs, doc(cfg(feature = "parquet")))]
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
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        Self::concat_impl(lfs, args)
    }

    /// Create a LazyFrame directly from a parquet scan.
    #[cfg_attr(docsrs, doc(cfg(feature = "parquet")))]
    pub fn scan_parquet(path: impl AsRef<Path>, args: ScanArgsParquet) -> PolarsResult<Self> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();
        if path_str.contains('*') {
            let paths = glob::glob(&path_str)
                .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                    Self::scan_parquet_impl(
                        path,
                        args.n_rows,
                        args.cache,
                        ParallelStrategy::None,
                        None,
                        args.rechunk,
                        args.low_memory,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?;

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
            )
        }
    }
}
