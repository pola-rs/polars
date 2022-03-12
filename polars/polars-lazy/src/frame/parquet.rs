use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::RowCount;

#[derive(Clone)]
pub struct ScanArgsParquet {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub parallel: bool,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
}

impl Default for ScanArgsParquet {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            parallel: true,
            rechunk: true,
            row_count: None,
        }
    }
}

impl LazyFrame {
    fn scan_parquet_impl(
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        parallel: bool,
        row_count: Option<RowCount>,
    ) -> Result<Self> {
        let mut lf: LazyFrame =
            LogicalPlanBuilder::scan_parquet(path, n_rows, cache, parallel, row_count)?
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
                .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                    let path_string = path.to_string_lossy().into_owned();
                    Self::scan_parquet_impl(path_string, args.n_rows, args.cache, false, None)
                })
                .collect::<Result<Vec<_>>>()?;

            concat(&lfs, args.rechunk)
                .map_err(|_| PolarsError::ComputeError("no matching files found".into()))
                .map(|mut lf| {
                    if let Some(n_rows) = args.n_rows {
                        lf = lf.slice(0, n_rows as u32)
                    };

                    if let Some(rc) = args.row_count {
                        lf = lf.with_row_count(&rc.name, Some(rc.offset))
                    };
                    lf
                })
        } else {
            Self::scan_parquet_impl(path, args.n_rows, args.cache, args.parallel, args.row_count)
        }
    }
}
