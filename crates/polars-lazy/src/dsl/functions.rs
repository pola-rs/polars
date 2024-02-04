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

/// Concat [LazyFrame]s horizontally.
pub fn concat_lf_horizontal<L: AsRef<[LazyFrame]>>(
    inputs: L,
    args: UnionArgs,
) -> PolarsResult<LazyFrame> {
    let lfs = inputs.as_ref();
    let mut opt_state = lfs.first().map(|lf| lf.opt_state).ok_or_else(
        || polars_err!(NoData: "Require at least one LazyFrame for horizontal concatenation"),
    )?;

    for lf in &lfs[1..] {
        // ensure we enable file caching if any lf has it enabled
        opt_state.file_caching |= lf.opt_state.file_caching;
    }

    let mut lps = Vec::with_capacity(lfs.len());
    let mut schemas = Vec::with_capacity(lfs.len());

    for lf in lfs.iter() {
        let mut lf = lf.clone();
        let schema = lf.schema()?;
        schemas.push(schema);
        let lp = std::mem::take(&mut lf.logical_plan);
        lps.push(lp);
    }

    let combined_schema = merge_schemas(&schemas)?;

    let options = HConcatOptions {
        parallel: args.parallel,
    };
    let lp = LogicalPlan::HConcat {
        inputs: lps,
        schema: Arc::new(combined_schema),
        options,
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

/** Concat multiple [`LazyFrame`]s vertically.

# Example
```
# use polars_core::prelude::*;
# use polars_lazy::prelude::*;

let df_v1 = df!(
        "a"=> &[1],
        "b"=> &[3],
)?;
let df_v2 = df!(
        "a"=> &[2],
        "b"=> &[4],
)?;
let df_vertical_concat = concat(
    [df_v1.lazy(), df_v2.lazy()],
    UnionArgs::default(),
)?
.collect()?;

assert_eq!(df_vertical_concat,
    df!(
        "a" => &[1,2],
        "b" => &[3,4],
    )?
);

# Ok::<(), PolarsError>(())
```
 **/
pub fn concat<L: AsRef<[LazyFrame]>>(inputs: L, args: UnionArgs) -> PolarsResult<LazyFrame> {
    concat_impl(
        inputs,
        args.rechunk,
        args.parallel,
        false,
        args.to_supertypes,
    )
}

/** Collect all [`LazyFrame`] computations.

# Example
```
# use polars_lazy::prelude::*;
# use polars_core::prelude::*;

let collected = collect_all([
    df!("a" => &[1,2])?.lazy(),
    df!("b" => &[3,4])?.lazy()
])?;

assert_eq!(collected,
    [
        df!("a" => &[1,2])?,
        df!("b" => &[3,4])?
    ]
);

# Ok::<(), PolarsError>(())
```
**/
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
