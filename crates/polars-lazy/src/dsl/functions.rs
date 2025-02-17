//! # Functions
//!
//! Function on multiple expressions.
//!

use polars_core::prelude::*;
pub use polars_plan::dsl::functions::*;
use polars_plan::prelude::UnionArgs;
use rayon::prelude::*;

use crate::prelude::*;

pub(crate) fn concat_impl<L: AsRef<[LazyFrame]>>(
    inputs: L,
    args: UnionArgs,
) -> PolarsResult<LazyFrame> {
    let mut inputs = inputs.as_ref().to_vec();

    let lf = std::mem::take(
        inputs
            .get_mut(0)
            .ok_or_else(|| polars_err!(NoData: "empty container given"))?,
    );

    let mut opt_state = lf.opt_state;
    let cached_arenas = lf.cached_arena.clone();

    let mut lps = Vec::with_capacity(inputs.len());
    lps.push(lf.logical_plan);

    for lf in &mut inputs[1..] {
        // Ensure we enable file caching if any lf has it enabled.
        if lf.opt_state.contains(OptFlags::FILE_CACHING) {
            opt_state |= OptFlags::FILE_CACHING;
        }
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp)
    }

    let lp = DslPlan::Union { inputs: lps, args };
    Ok(LazyFrame::from_inner(lp, opt_state, cached_arenas))
}

#[cfg(feature = "diagonal_concat")]
/// Concat [LazyFrame]s diagonally.
/// Calls [`concat`][concat()] internally.
pub fn concat_lf_diagonal<L: AsRef<[LazyFrame]>>(
    inputs: L,
    mut args: UnionArgs,
) -> PolarsResult<LazyFrame> {
    args.diagonal = true;
    concat_impl(inputs, args)
}

/// Concat [LazyFrame]s horizontally.
pub fn concat_lf_horizontal<L: AsRef<[LazyFrame]>>(
    inputs: L,
    args: UnionArgs,
) -> PolarsResult<LazyFrame> {
    let lfs = inputs.as_ref();
    let (mut opt_state, cached_arena) = lfs
        .first()
        .map(|lf| (lf.opt_state, lf.cached_arena.clone()))
        .ok_or_else(
            || polars_err!(NoData: "Require at least one LazyFrame for horizontal concatenation"),
        )?;

    for lf in &lfs[1..] {
        // Ensure we enable file caching if any lf has it enabled.
        if lf.opt_state.contains(OptFlags::FILE_CACHING) {
            opt_state |= OptFlags::FILE_CACHING;
        }
    }

    let options = HConcatOptions {
        parallel: args.parallel,
    };
    let lp = DslPlan::HConcat {
        inputs: lfs.iter().map(|lf| lf.logical_plan.clone()).collect(),
        options,
    };
    Ok(LazyFrame::from_inner(lp, opt_state, cached_arena))
}

/// Concat multiple [`LazyFrame`]s vertically.
pub fn concat<L: AsRef<[LazyFrame]>>(inputs: L, args: UnionArgs) -> PolarsResult<LazyFrame> {
    concat_impl(inputs, args)
}

/// Collect all [`LazyFrame`] computations.
pub fn collect_all<I>(lfs: I) -> PolarsResult<Vec<DataFrame>>
where
    I: IntoParallelIterator<Item = LazyFrame>,
{
    let iter = lfs.into_par_iter();

    polars_core::POOL.install(|| iter.map(|lf| lf.collect()).collect())
}

#[cfg(test)]
mod test {
    // used only if feature="diagonal_concat"
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "diagonal_concat")]
    fn test_diag_concat_lf() -> PolarsResult<()> {
        let a = df![
            "a" => [1, 2],
            "b" => ["a", "b"]
        ]?;

        let b = df![
            "b" => ["a", "b"],
            "c" => [1, 2]
        ]?;

        let c = df![
            "a" => [5, 7],
            "c" => [1, 2],
            "d" => [1, 2]
        ]?;

        let out = concat_lf_diagonal(
            &[a.lazy(), b.lazy(), c.lazy()],
            UnionArgs {
                rechunk: false,
                parallel: false,
                ..Default::default()
            },
        )?
        .collect()?;

        let expected = df![
            "a" => [Some(1), Some(2), None, None, Some(5), Some(7)],
            "b" => [Some("a"), Some("b"), Some("a"), Some("b"), None, None],
            "c" => [None, None, Some(1), Some(2), Some(1), Some(2)],
            "d" => [None, None, None, None, Some(1), Some(2)]
        ]?;

        assert!(out.equals_missing(&expected));

        Ok(())
    }
}
