use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::frame::join::*;
use crate::prelude::*;
use crate::series::ops::coalesce_series;

pub fn replace(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    let output_dtype = try_get_supertype(s.dtype(), new.dtype())?;

    if old.len() == 0 {
        return s.cast(&output_dtype);
    }

    // TODO: Add fast path for replacing a single value (ZIP WITH?)
    let replaced = join_replacer(s, old, new)?;

    coalesce_series(&[replaced, s.cast(&output_dtype)?])
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

    // TODO: Add fast path for replacing a single value
    let replaced = join_replacer(s, old, new)?;

    coalesce_series(&[replaced, default])
}

/// Create a Series containing only the replaced values at the right indices
/// and nulls everywhere else
fn join_replacer(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    // length 1 is many-to-one replace, otherwise it's one-to-one
    polars_ensure!(
        (new.len() == old.len()) || new.len() == 1,
        ComputeError: "`new` input for `replace` must have the same length as `old` or have length 1"
    );

    let join_dtype = try_get_supertype(s.dtype(), old.dtype())?;

    let df = DataFrame::new_no_checks(vec![s.cast(&join_dtype)?]);

    let mut old = old.cast(&join_dtype)?;
    old.rename("__POLARS_REPLACE_OLD");
    let mut new = match new.len() {
        1 => new.new_from_index(0, old.len()),
        _ => new.clone(),
    };
    new.rename("__POLARS_REPLACE_NEW");
    let replacer = DataFrame::new_no_checks(vec![old, new]);

    let df_joined = df.join(
        &replacer,
        [s.name()],
        ["__POLARS_REPLACE_OLD"],
        JoinArgs {
            how: JoinType::Left,
            join_nulls: true,
            ..Default::default()
        },
    )?;

    let s_joined = df_joined.column("__POLARS_REPLACE_NEW").unwrap();
    Ok(s_joined.clone())
}
