use crate::functions::concat;
use crate::prelude::*;
use polars_core::prelude::*;

#[derive(Copy, Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: bool,
    pub rechunk: bool,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            parallel: true,
            rechunk: true,
        }
    }
}

impl LazyFrame {
    fn scan_parquet_impl(
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        parallel: bool,
    ) -> Result<Self> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(path, n_rows, cache, parallel)?
            .build()
            .into();
        lf.opt_state.agg_scan_projection = true;
        Ok(lf)
    }

    /// Create a LazyFrame directly from a parquet scan.
    #[cfg_attr(docsrs, doc(cfg(feature = "parquet")))]
    pub fn scan_parquet(path: String, args: ScanArgsParquet) -> Result<Self> {
        if path.contains('*') {
            let paths = glob::glob(&path)
                .map_err(|_| PolarsError::ValueError("invalid glob pattern given".into()))?;
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                    let path_string = path.to_string_lossy().into_owned();
                    Self::scan_parquet_impl(path_string, args.n_rows, args.cache, false)
                })
                .collect::<Result<Vec<_>>>()?;

            concat(&lfs, args.rechunk)
                .map_err(|_| PolarsError::ComputeError("no matching files found".into()))
                .map(|lf| {
                    if let Some(n_rows) = args.n_rows {
                        lf.slice(0, n_rows as u32)
                    } else {
                        lf
                    }
                })
        } else {
            Self::scan_parquet_impl(path, args.n_rows, args.cache, args.parallel)
        }
    }
}
