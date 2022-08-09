use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::parquet::ParallelStrategy;
use polars_io::RowCount;

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
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        parallel: ParallelStrategy,
        row_count: Option<RowCount>,
        rechunk: bool,
        low_memory: bool,
    ) -> Result<Self> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_parquet(
            path, n_rows, cache, parallel, None, rechunk, low_memory,
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

    fn concat_impl(lfs: Vec<LazyFrame>, args: ScanArgsParquet) -> Result<LazyFrame> {
        concat(&lfs, args.rechunk).map(|mut lf| {
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
    pub fn scan_parquet_files(paths: Vec<String>, args: ScanArgsParquet) -> Result<Self> {
        let lfs = paths
            .iter()
            .map(|p| {
                Self::scan_parquet_impl(
                    p.to_string(),
                    args.n_rows,
                    args.cache,
                    args.parallel,
                    None,
                    args.rechunk,
                    args.low_memory,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Self::concat_impl(lfs, args)
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
                    Self::scan_parquet_impl(
                        path_string,
                        args.n_rows,
                        args.cache,
                        ParallelStrategy::None,
                        None,
                        args.rechunk,
                        args.low_memory,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

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
