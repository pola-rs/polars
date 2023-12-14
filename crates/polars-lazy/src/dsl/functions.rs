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
    convert_supertypes: bool,
) -> PolarsResult<LazyFrame> {
    let mut inputs = inputs.as_ref().to_vec();

    let mut lf = std::mem::take(
        inputs
            .get_mut(0)
            .ok_or_else(|| polars_err!(NoData: "empty container given"))?,
    );

    let mut opt_state = lf.opt_state;
    let options = UnionOptions {
        parallel,
        from_partitioned_ds,
        rechunk,
        ..Default::default()
    };

    let lf = match &mut lf.logical_plan {
        // re-use the same union
        LogicalPlan::Union {
            inputs: existing_inputs,
            options: opts,
        } if opts == &options => {
            for lf in &mut inputs[1..] {
                // ensure we enable file caching if any lf has it enabled
                opt_state.file_caching |= lf.opt_state.file_caching;
                let lp = std::mem::take(&mut lf.logical_plan);
                existing_inputs.push(lp)
            }
            lf
        },
        _ => {
            let mut lps = Vec::with_capacity(inputs.len());
            lps.push(lf.logical_plan);

            for lf in &mut inputs[1..] {
                // ensure we enable file caching if any lf has it enabled
                opt_state.file_caching |= lf.opt_state.file_caching;
                let lp = std::mem::take(&mut lf.logical_plan);
                lps.push(lp)
            }

            let lp = LogicalPlan::Union {
                inputs: lps,
                options,
            };
            let mut lf = LazyFrame::from(lp);
            lf.opt_state = opt_state;

            lf
        },
    };

    if convert_supertypes {
        let LogicalPlan::Union {
            mut inputs,
            options,
        } = lf.logical_plan
        else {
            unreachable!()
        };
        let mut schema = inputs[0].schema()?.as_ref().as_ref().clone();

        let mut changed = false;
        for input in inputs[1..].iter() {
            changed |= schema.to_supertype(input.schema()?.as_ref().as_ref())?;
        }

        let mut placeholder = LogicalPlan::default();
        if changed {
            let mut exprs = vec![];
            for input in &mut inputs {
                std::mem::swap(input, &mut placeholder);
                let input_schema = placeholder.schema()?;

                exprs.clear();
                let to_cast = input_schema.iter().zip(schema.iter_dtypes()).flat_map(
                    |((left_name, left_type), st)| {
                        if left_type != st {
                            Some(col(left_name.as_ref()).cast(st.clone()))
                        } else {
                            None
                        }
                    },
                );
                exprs.extend(to_cast);
                let mut lf = LazyFrame::from(placeholder);
                if !exprs.is_empty() {
                    lf = lf.with_columns(exprs.as_slice());
                }

                placeholder = lf.logical_plan;
                std::mem::swap(&mut placeholder, input);
            }
        }
        Ok(LazyFrame::from(LogicalPlan::Union { inputs, options }))
    } else {
        Ok(lf)
    }
}

#[cfg(feature = "diagonal_concat")]
/// Concat [LazyFrame]s diagonally.
/// Calls [`concat`][concat()] internally.
pub fn concat_lf_diagonal<L: AsRef<[LazyFrame]>>(
    inputs: L,
    args: UnionArgs,
) -> PolarsResult<LazyFrame> {
    let lfs = inputs.as_ref();
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
        .iter()
        // Zip Frames with their Schemas
        .zip(schemas)
        .map(|(lf, lf_schema)| {
            let mut lf = lf.clone();
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

    concat(lfs_with_all_columns, args)
}

#[cfg(feature = "horizontal_concat")]
/// Concat [LazyFrame]s horizontally.
pub fn concat_lf_horizontal<L: AsRef<[LazyFrame]>>(inputs: L) -> PolarsResult<LazyFrame> {
    let lfs = inputs.as_ref();
    let mut opt_state = lfs.first().map(|lf| lf.opt_state).ok_or_else(
        || polars_err!(NoData: "Require at least one LazyFrame for horizontal concatenation"),
    )?;

    for lf in &lfs[1..] {
        // ensure we enable file caching if any lf has it enabled
        opt_state.file_caching |= lf.opt_state.file_caching;
    }

    let schema_size = lfs
        .iter()
        .map(|lf| lf.schema().map(|schema| schema.len()))
        .sum::<PolarsResult<_>>()?;
    let mut column_names = PlHashSet::with_capacity(schema_size);
    let mut combined_schema = Schema::with_capacity(schema_size);

    let mut lps = Vec::with_capacity(lfs.len());

    for lf in lfs.iter() {
        let mut lf = lf.clone();
        let schema = lf.schema()?;
        schema.iter().try_for_each(|(name, dtype)| {
            if !column_names.contains(name) {
                column_names.insert(name.clone());
                combined_schema.with_column(name.clone(), dtype.clone());
                Ok(())
            } else {
               Err(polars_err!(Duplicate: "Column with name '{}' has more than one occurrence", name))
            }
        })?;
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp);
    }

    let lp = LogicalPlan::HConcat {
        inputs: lps,
        schema: Arc::new(combined_schema),
    };
    let mut lf = LazyFrame::from(lp);
    lf.opt_state = opt_state;

    Ok(lf)
}

#[derive(Clone, Copy)]
pub struct UnionArgs {
    pub parallel: bool,
    pub rechunk: bool,
    pub to_supertypes: bool,
}

impl Default for UnionArgs {
    fn default() -> Self {
        Self {
            parallel: true,
            rechunk: true,
            to_supertypes: false,
        }
    }
}

/// Concat multiple [`LazyFrame`]s vertically.
pub fn concat<L: AsRef<[LazyFrame]>>(inputs: L, args: UnionArgs) -> PolarsResult<LazyFrame> {
    concat_impl(
        inputs,
        args.rechunk,
        args.parallel,
        false,
        args.to_supertypes,
    )
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

    #[test]
    #[cfg(feature = "horizontal_concat")]
    fn test_horizontal_concat_lf() -> PolarsResult<()> {
        let a = df![
            "a1" => [1, 2, 3],
            "a2" => ["a", "b", "c"]
        ]?;

        let b = df![
            "b1" => [0.25, 0.5],
        ]?;

        let c = df![
            "c1" => [1, 2, 3, 4],
            "c2" => [5, 6, 7, 8],
            "c3" => [9, 10, 11, 12]
        ]?;

        let out = concat_lf_horizontal(&[a.lazy(), b.lazy(), c.lazy()])?.collect()?;

        let expected = df![
            "a1" => [Some(1), Some(2), Some(3), None],
            "a2" => [Some("a"), Some("b"), Some("c"), None],
            "b1" => [Some(0.25), Some(0.5), None, None],
            "c1" => [Some(1), Some(2), Some(3), Some(4)],
            "c2" => [Some(5), Some(6), Some(7), Some(8)],
            "c3" => [Some(9), Some(10), Some(11), Some(12)],
        ]?;

        assert!(out.equals_missing(&expected));

        Ok(())
    }
}
