//! # Functions
//!
//! Function on multiple expressions.
//!
use polars_core::prelude::*;
pub use polars_plan::dsl::functions::*;
use rayon::prelude::*;

use crate::prelude::*;

pub(crate) fn concat_impl<L: AsRef<[LazyFrame]>>(
    inputs: L,
    rechunk: bool,
    parallel: bool,
    from_partitioned_ds: bool,
) -> PolarsResult<LazyFrame> {
    let mut inputs = inputs.as_ref().to_vec();
    let lf = std::mem::take(
        inputs
            .get_mut(0)
            .ok_or_else(|| PolarsError::ComputeError("empty container given".into()))?,
    );
    let mut opt_state = lf.opt_state;
    let mut lps = Vec::with_capacity(inputs.len());
    lps.push(lf.logical_plan);

    for lf in &mut inputs[1..] {
        // ensure we enable file caching if any lf has it enabled
        opt_state.file_caching |= lf.opt_state.file_caching;
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp)
    }
    let options = UnionOptions {
        parallel,
        from_partitioned_ds,
        ..Default::default()
    };

    let lp = LogicalPlan::Union {
        inputs: lps,
        options,
    };
    let mut lf = LazyFrame::from(lp);
    lf.opt_state = opt_state;

    if rechunk {
        Ok(lf.map_private(FunctionNode::Rechunk))
    } else {
        Ok(lf)
    }
}

/// Concat multiple
pub fn concat<L: AsRef<[LazyFrame]>>(
    inputs: L,
    rechunk: bool,
    parallel: bool,
) -> PolarsResult<LazyFrame> {
    concat_impl(inputs, rechunk, parallel, false)
}

/// Collect all `LazyFrame` computations.
pub fn collect_all<I>(lfs: I) -> PolarsResult<Vec<DataFrame>>
where
    I: IntoParallelIterator<Item = LazyFrame>,
{
    let iter = lfs.into_par_iter();

    polars_core::POOL.install(|| iter.map(|lf| lf.collect()).collect())
}
