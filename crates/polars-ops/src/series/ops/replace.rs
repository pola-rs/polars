use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::frame::join::*;
use crate::prelude::*;
use crate::series::is_in;

pub fn replace(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
    return_dtype: Option<DataType>,
) -> PolarsResult<Series> {
    let return_dtype = match return_dtype {
        Some(dtype) => dtype,
        None => try_get_supertype(new.dtype(), default.dtype())?,
    };

    let default = match default.len() {
        len if len == s.len() => default.cast(&return_dtype)?,
        1 => default.cast(&return_dtype)?.new_from_index(0, s.len()),
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

    let old = match (s.dtype(), old.dtype()) {
        (s_dt, old_dt) if s_dt == old_dt => old.clone(),
        (DataType::Categorical(opt_rev_map, ord), DataType::Utf8) => {
            let dt = opt_rev_map
                .as_ref()
                .filter(|rev_map| rev_map.is_enum())
                .map(|rev_map| DataType::Categorical(Some(rev_map.clone()), *ord))
                .unwrap_or(DataType::Categorical(None, *ord));

            old.strict_cast(&dt)?
        },
        _ => old.strict_cast(s.dtype())?,
    };
    let new = new.cast(&return_dtype)?;

    if new.len() == 1 {
        replace_by_single(s, &old, &new, &default)
    } else {
        replace_by_multiple(s, old, new, &default)
    }
}

// Fast path for replacing by a single value
fn replace_by_single(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    let mask = is_in(s, old)?;
    let new_broadcast = new.new_from_index(0, default.len());
    new_broadcast.zip_with(&mask, default)
}

/// General case for replacing by multiple values
fn replace_by_multiple(
    s: &Series,
    old: Series,
    new: Series,
    default: &Series,
) -> PolarsResult<Series> {
    polars_ensure!(
        new.len() == old.len(),
        ComputeError: "`new` input for `replace` must have the same length as `old` or have length 1"
    );

    let df = DataFrame::new_no_checks(vec![s.clone()]);
    let replacer = create_replacer(old, new)?;

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

    let replaced = joined.column("__POLARS_REPLACE_NEW").unwrap();

    match joined.column("__POLARS_REPLACE_MASK") {
        Ok(col) => {
            let mask = col.bool()?;
            replaced.zip_with(mask, default)
        },
        Err(_) => {
            if replaced.null_count() > 0 {
                let mask = &replaced.is_not_null();
                replaced.zip_with(mask, default)
            } else {
                Ok(replaced.clone())
            }
        },
    }
}

// Build replacer dataframe
fn create_replacer(mut old: Series, mut new: Series) -> PolarsResult<DataFrame> {
    old.rename("__POLARS_REPLACE_OLD");
    new.rename("__POLARS_REPLACE_NEW");

    let cols = if new.null_count() > 0 {
        let mask = Series::new("__POLARS_REPLACE_MASK", &[true]).new_from_index(0, new.len());
        vec![old, new, mask]
    } else {
        vec![old, new]
    };
    let out = DataFrame::new_no_checks(cols);
    Ok(out)
}
