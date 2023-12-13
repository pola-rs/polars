use polars_core::prelude::arity::binary_elementwise;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::frame::join::*;
use crate::prelude::*;
use crate::series::is_in;

pub fn replace(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    if old.len() == 0 {
        let output_dtype = try_get_supertype(s.dtype(), new.dtype())?;
        return s.cast(&output_dtype);
    }
    if new.len() == 1 {
        return replace_many_to_one(s, old, new, s);
    }

    let replaced = get_replaced(s, old, new)?;
    coalesce_replaced(&replaced, s, s, old, new)
}

pub fn replace_with_default(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    let output_dtype = try_get_supertype(new.dtype(), default.dtype())?;

    let default = match default.len() {
        len if len == s.len() => default.cast(&output_dtype)?,
        1 => default.cast(&output_dtype)?.new_from_index(0, s.len()),
        _ => {
            polars_bail!(
                ComputeError:
                "`default` input for `replace` must have the same length as the input or have length 1"
            )
        },
    };

    if old.len() == 0 {
        return Ok(default);
    }
    if new.len() == 1 {
        return replace_many_to_one(s, old, new, &default);
    }

    let replaced = get_replaced(s, old, new)?;
    coalesce_replaced(&replaced, &default, s, old, new)
}

// Fast path for replacing by a single value
fn replace_many_to_one(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    let condition = is_in(s, old)?;
    let new = new.new_from_index(0, default.len());
    new.zip_with(&condition, default)
}

/// Create a Series containing only the replaced values and nulls everywhere else.
fn get_replaced(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    // length 1 is many-to-one replace, otherwise it's one-to-one
    polars_ensure!(
        (new.len() == old.len()) || new.len() == 1,
        ComputeError: "`new` input for `replace` must have the same length as `old` or have length 1"
    );

    let df = DataFrame::new_no_checks(vec![s.clone()]);

    let mut old = if old.dtype() == s.dtype() {
        old.clone()
    } else {
        old.cast(s.dtype())?
    };
    let mut new = new.clone();
    old.rename("__POLARS_REPLACE_OLD");
    new.rename("__POLARS_REPLACE_NEW");
    let replacer = DataFrame::new_no_checks(vec![old, new]);

    let joined = df.join(
        &replacer,
        [s.name()],
        ["__POLARS_REPLACE_OLD"],
        JoinArgs {
            how: JoinType::Left,
            join_nulls: true,
            ..Default::default()
        },
    )?;

    let out = joined.column("__POLARS_REPLACE_NEW").unwrap().clone();
    Ok(out)
}

/// Coalesce the replaced values with another column to get the final result.
fn coalesce_replaced(
    replaced: &Series,
    other: &Series,
    s: &Series,
    old: &Series,
    new: &Series,
) -> PolarsResult<Series> {
    let mask = match new.null_count() {
        0 => replaced.is_not_null(),
        _ => {
            //  If we replace some values by null, we cannot do a regular coalesce
            let is_not_null = replaced.is_not_null();

            let null_keys = determine_null_keys(old, new).unwrap();
            let mapped_to_null = is_in(s, &null_keys)?;

            let opt_or = |l: Option<bool>, r: Option<bool>| {
                let l = l.unwrap();
                match r {
                    Some(r) => l | r,
                    None => false,
                }
            };
            binary_elementwise(&is_not_null, &mapped_to_null, opt_or)
        },
    };

    let out = replaced.zip_with(&mask, other)?;
    Ok(out)
}

/// Determine the keys that map to null values
fn determine_null_keys(old: &Series, new: &Series) -> PolarsResult<Series> {
    old.filter(&new.is_null())
}
