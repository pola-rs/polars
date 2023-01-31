use std::path::Path;

use polars_core::prelude::*;
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsIpc {
    pub n_rows: Option<usize>,
    pub cache: bool,
    pub rechunk: bool,
    pub row_count: Option<RowCount>,
    pub memmap: bool,
}

impl Default for ScanArgsIpc {
    fn default() -> Self {
        Self {
            n_rows: None,
            cache: true,
            rechunk: true,
            row_count: None,
            memmap: true,
        }
    }
}

impl LazyFrame {
    fn scan_ipc_impl(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        let options = IpcScanOptions {
            n_rows: args.n_rows,
            cache: args.cache,
            with_columns: None,
            row_count: None,
            rechunk: args.rechunk,
            memmap: args.memmap,
        };
        let row_count = args.row_count;
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(path.as_ref(), options)?
            .build()
            .into();
        lf.opt_state.file_caching = true;

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }

        Ok(lf)
    }

    /// Create a LazyFrame directly from a ipc scan.
    pub fn scan_ipc(path: impl AsRef<Path>, args: ScanArgsIpc) -> PolarsResult<Self> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy();
        if path_str.contains('*') {
            let paths = glob::glob(&path_str)
                .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;

            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
                    let mut args = args.clone();
                    args.rechunk = false;
                    args.row_count = None;
                    Self::scan_ipc_impl(path, args)
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            concat_impl(&lfs, args.rechunk, true, true)
                .map_err(|_| PolarsError::ComputeError("no matching files found".into()))
                .map(|mut lf| {
                    if let Some(n_rows) = args.n_rows {
                        lf = lf.slice(0, n_rows as IdxSize);
                    };

                    if let Some(rc) = args.row_count {
                        lf = lf.with_row_count(&rc.name, Some(rc.offset))
                    }

                    lf
                })
        } else {
            Self::scan_ipc_impl(path, args)
        }
    }
}
