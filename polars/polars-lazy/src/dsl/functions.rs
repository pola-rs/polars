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

#[cfg(feature = "diagonal_concat")]
/// Concat [LazyFrame]s diagonally.
/// Calls [concat] internally.
pub fn diag_concat_lf<L: AsRef<[LazyFrame]>>(
    lfs: L,
    rechunk: bool,
    parallel: bool,
) -> PolarsResult<LazyFrame> {
    let lfs = lfs.as_ref().to_vec();
    let schemas = lfs
        .iter()
        .map(|lf| lf.schema())
        .collect::<PolarsResult<Vec<_>>>()?;

    let upper_bound_width = schemas.iter().map(|sch| sch.len()).sum();

    // Use Vec to preserve order
    let mut column_names = Vec::with_capacity(upper_bound_width);
    let mut total_schema = Vec::with_capacity(upper_bound_width);

    for sch in schemas.iter() {
        sch.iter().for_each(|(name, dtype)| {
            if !column_names.contains(name) {
                column_names.push(name.clone());
                total_schema.push((name.clone(), dtype.clone()));
            }
        });
    }

    let lfs_with_all_columns = lfs
        .into_iter()
        // Zip Frames with their Schemas
        .zip(schemas.into_iter())
        .map(|(mut lf, lf_schema)| {
            for (name, dtype) in total_schema.iter() {
                // If a name from Total Schema is not present - append
                if lf_schema.get_field(name).is_none() {
                    lf = lf.with_column(NULL.lit().cast(dtype.clone()).alias(name))
                }
            }

            // Now, reorder to match schema
            let reordered_lf = lf.select(
                column_names
                    .iter()
                    .map(|col_name| col(col_name))
                    .collect::<Vec<Expr>>(),
            );

            Ok(reordered_lf)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    concat(lfs_with_all_columns, rechunk, parallel)
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

        let out = diag_concat_lf(&[a.lazy(), b.lazy(), c.lazy()], false, false)?.collect()?;

        let expected = df![
            "a" => [Some(1), Some(2), None, None, Some(5), Some(7)],
            "b" => [Some("a"), Some("b"), Some("a"), Some("b"), None, None],
            "c" => [None, None, Some(1), Some(2), Some(1), Some(2)],
            "d" => [None, None, None, None, Some(1), Some(2)]
        ]?;

        assert!(out.frame_equal_missing(&expected));

        Ok(())
    }
}
