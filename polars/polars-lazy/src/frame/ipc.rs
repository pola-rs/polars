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
    /// Create a LazyFrame directly from a ipc scan.
    #[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
    pub fn scan_ipc(path: String, args: ScanArgsIpc) -> Result<Self> {
        let options = LpScanOptions {
            n_rows: args.n_rows,
            cache: args.cache,
            with_columns: None,
        };
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(path, options)?.build().into();
        lf.opt_state.agg_scan_projection = true;
        Ok(lf)
    }
}
