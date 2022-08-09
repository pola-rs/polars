use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::RowCount;

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
    fn scan_ipc_impl(path: String, args: ScanArgsIpc) -> Result<Self> {
        let options = IpcScanOptions {
            n_rows: args.n_rows,
            cache: args.cache,
            with_columns: None,
            row_count: None,
            rechunk: args.rechunk,
            memmap: args.memmap,
        };
        let row_count = args.row_count;
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_ipc(path, options)?.build().into();
        lf.opt_state.file_caching = true;

        // it is a bit hacky, but this row_count function updates the schema
        if let Some(row_count) = row_count {
            lf = lf.with_row_count(&row_count.name, Some(row_count.offset))
        }

        Ok(lf)
    }

    /// Create a LazyFrame directly from a ipc scan.
    #[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
    pub fn scan_ipc(path: String, args: ScanArgsIpc) -> Result<Self> {
        if path.contains('*') {
            let paths = glob::glob(&path)
                .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;
            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                    let path_string = path.to_string_lossy().into_owned();
                    let mut args = args.clone();
                    args.row_count = None;
                    Self::scan_ipc_impl(path_string, args)
                })
                .collect::<Result<Vec<_>>>()?;

            concat(&lfs, args.rechunk)
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
