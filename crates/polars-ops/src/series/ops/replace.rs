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

    let (replaced, mask) = get_replaced(s, old, new)?;
    coalesce_replaced(&replaced, s, &mask)
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

    let (replaced, mask) = get_replaced(s, old, new)?;
    coalesce_replaced(&replaced, &default, &mask)
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
fn get_replaced(s: &Series, old: &Series, new: &Series) -> PolarsResult<(Series, Series)> {
    // length 1 is many-to-one replace, otherwise it's one-to-one
    polars_ensure!(
        (new.len() == old.len()) || new.len() == 1,
        ComputeError: "`new` input for `replace` must have the same length as `old` or have length 1"
    );

    let df = DataFrame::new_no_checks(vec![s.clone()]);

    // Build replacer dataframe
    let mut old = if old.dtype() == s.dtype() {
        old.clone()
    } else {
        old.cast(s.dtype())?
    };
    old.rename("__POLARS_REPLACE_OLD");

    let mut new = new.clone();
    new.rename("__POLARS_REPLACE_NEW");

    let replacer = if new.null_count() > 0 {
        // If we replace some values by null, we need to track which values were replaced
        let mask = Series::new("__POLARS_REPLACE_MASK", &[true]).new_from_index(0, new.len());
        DataFrame::new_no_checks(vec![old, new, mask])
    } else {
        DataFrame::new_no_checks(vec![old, new])
    };

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

    let replaced = joined.column("__POLARS_REPLACE_NEW").unwrap().clone();
    let mask = match joined.column("__POLARS_REPLACE_MASK").ok() {
        Some(col) => col.clone(),
        None => replaced.is_not_null().into_series(),
    };

    Ok((replaced, mask))
}

/// Coalesce the replaced values with another column to get the final result.
fn coalesce_replaced(replaced: &Series, other: &Series, mask: &Series) -> PolarsResult<Series> {
    replaced.zip_with(mask.bool()?, other)
}
