use crate::functions::concat;
use crate::prelude::*;
use polars_core::prelude::*;

#[derive(Copy, Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: true,
        }
    }
}

impl LazyFrame {
    fn scan_ipc_impl(path: String, args: ScanArgsIpc) -> Result<Self> {
        let options = LpScanOptions {
            n_rows: args.n_rows,
            cache: args.cache,
            with_columns: None,
        };
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(path, options)?.build().into();
        lf.opt_state.agg_scan_projection = true;
        Ok(lf)
    }

    /// Create a LazyFrame directly from a ipc scan.
    #[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
    pub fn scan_ipc(path: String, args: ScanArgsIpc) -> Result<Self> {
        if path.contains('*') {
            let paths = glob::glob(&path)
                .map_err(|_| PolarsError::ValueError("invalid glob pattern given".into()))?;
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                    let path_string = path.to_string_lossy().into_owned();
                    Self::scan_ipc_impl(path_string, args)
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
            Self::scan_ipc_impl(path, args)
        }
    }
}
