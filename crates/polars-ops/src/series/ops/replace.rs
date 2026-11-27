use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_defs::join::{JoinArgs, JoinCoalesce, JoinType};
use polars_error::polars_ensure;

use crate::frame::join::*;
use crate::prelude::*;

fn find_output_length(
    fnname: &str,
    items: impl IntoIterator<Item = (&'static str, usize)>,
) -> PolarsResult<usize> {
    let mut length = 1;
    for (argument_idx, (argument, l)) in items.into_iter().enumerate() {
        if l != 1 {
            if l != length && length != 1 {
                polars_bail!(
                    length_mismatch = fnname,
                    l,
                    length,
                    argument = argument,
                    argument_idx = argument_idx
                );
            }
            length = l;
        }
    }
    Ok(length)
}

/// Replace values by different values of the same data type.
pub fn replace(s: &Series, old: &ListChunked, new: &ListChunked) -> PolarsResult<Series> {
    find_output_length(
        "replace",
        [("self", s.len()), ("old", old.len()), ("new", new.len())],
    )?;

    polars_ensure!(
        old.len() == 1 && new.len() == 1,
        nyi = "`replace` with a replacement pattern per row"
    );

    let old = old.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let new = new.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;

    if old.is_empty() {
        return Ok(s.clone());
    }
    validate_old(&old)?;

    let dtype = s.dtype();
    let (old, kept) = cast_to_match(&old, dtype)?;
    let new = filter_new(new, kept.as_ref())?;
    if old.is_empty() {
        return Ok(s.clone());
    }
    let new = new.strict_cast(dtype)?;

    if new.len() == 1 {
        replace_by_single(s, &old, &new, s)
    } else {
        replace_by_multiple(s, old, new, s)
    }
}

/// Replace all values by different values.
///
/// Unmatched values are replaced by a default value.
pub fn replace_or_default(
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    default: &Series,
    return_dtype: Option<DataType>,
) -> PolarsResult<Series> {
    find_output_length(
        "replace_strict",
        [
            ("self", s.len()),
            ("old", old.len()),
            ("new", new.len()),
            ("default", default.len()),
        ],
    )?;

    polars_ensure!(
        old.len() == 1 && new.len() == 1,
        nyi = "`replace_strict` with a replacement pattern per row"
    );

    let old = old.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let new = new.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;

    polars_ensure!(
        default.len() == s.len() || default.len() == 1,
        InvalidOperation: "`default` input for `replace_strict` must have the same length as the input or have length 1"
    );
    validate_old(&old)?;

    let return_dtype = match return_dtype {
        Some(dtype) => dtype,
        None => try_get_supertype(new.dtype(), default.dtype())?,
    };
    let default = default.cast(&return_dtype)?;

    if old.is_empty() {
        return default.broadcast_owned_to(s.len());
    }

    let (old, kept) = cast_to_match(&old, s.dtype())?;
    let new = filter_new(new, kept.as_ref())?;
    if old.is_empty() {
        return default.broadcast_owned_to(s.len());
    }
    let new = new.cast(&return_dtype)?;

    if new.len() == 1 {
        replace_by_single(s, &old, &new, &default)
    } else {
        replace_by_multiple(s, old, new, &default)
    }
}

/// Replace all values by different values.
///
/// Raises an error if not all values were replaced.
pub fn replace_strict(
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    return_dtype: Option<DataType>,
) -> PolarsResult<Series> {
    find_output_length(
        "replace_strict",
        [("self", s.len()), ("old", old.len()), ("new", new.len())],
    )?;

    polars_ensure!(
        old.len() == 1 && new.len() == 1,
        nyi = "`replace_strict` with a replacement pattern per row"
    );

    let old = old.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let new = new.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;

    if old.is_empty() {
        polars_ensure!(
            s.len() == s.null_count(),
            InvalidOperation: "must specify which values to replace"
        );
        return Ok(s.clone());
    }
    validate_old(&old)?;

    // Extra check because strict_cast is too permissive, e.g. allows string -> struct cast.
    if old.dtype().can_cast_to(s.dtype()) != Some(true) {
        polars_bail!(
            InvalidOperation: "cannot use values of type `{}` to replace values in a column of type `{}`",
            old.dtype(),
            s.dtype()
        )
    }

    let (old, kept) = cast_to_match(&old, s.dtype())?;
    let new = filter_new(new, kept.as_ref())?;
    if old.is_empty() {
        // No `old` value can match, so all values are unmatched (a null mask): only nulls pass.
        let mask = BooleanChunked::full_null(PlSmallStr::EMPTY, s.len());
        ensure_all_replaced(&mask, s, false, false)?;
        let dtype = return_dtype.unwrap_or_else(|| new.dtype().clone());
        return Ok(Series::full_null(s.name().clone(), s.len(), &dtype));
    }

    let new = match return_dtype {
        Some(dtype) => new.strict_cast(&dtype)?,
        None => new,
    };

    if new.len() == 1 {
        replace_by_single_strict(s, &old, &new)
    } else {
        replace_by_multiple_strict(s, old, new)
    }
}

/// Validate the `old` input.
fn validate_old(old: &Series) -> PolarsResult<()> {
    polars_ensure!(
        old.n_unique()? == old.len(),
        InvalidOperation: "`old` input for `replace` must not contain duplicates"
    );
    Ok(())
}

// Fast path for replacing by a single value
fn replace_by_single(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    let mask = get_replacement_mask(s, old)?;
    new.zip_with(&mask, default)
}

/// Fast path for replacing by a single value in strict mode
fn replace_by_single_strict(s: &Series, old: &Series, new: &Series) -> PolarsResult<Series> {
    let mask = get_replacement_mask(s, old)?;
    ensure_all_replaced(&mask, s, old.null_count() > 0, true)?;

    let mut out = new.new_from_index(0, s.len());

    // Transfer validity from `mask` to `out`.
    if mask.null_count() > 0 {
        out = out.zip_with(&mask, &Series::new_null(PlSmallStr::EMPTY, s.len()))?
    }
    Ok(out)
}

/// Get a boolean mask of which values in the original Series will be replaced.
///
/// Null values are propagated to the mask, unless `old` contains a null.
fn get_replacement_mask(s: &Series, old: &Series) -> PolarsResult<BooleanChunked> {
    if old.null_count() == old.len() {
        // Fast path for when users are using `replace(None, ...)` instead of `fill_null`.
        Ok(s.is_null())
    } else if old.len() == 1 {
        Ok(s.equal(old)?)
    } else {
        let nulls_equal = old.null_count() > 0;
        let old = old.implode()?;
        is_in(s, &old.into_series(), nulls_equal)
    }
}

/// General case for replacing by multiple values
fn replace_by_multiple(
    s: &Series,
    old: Series,
    new: Series,
    default: &Series,
) -> PolarsResult<Series> {
    validate_new(&new, old.len())?;

    let df = s.clone().into_frame();
    let add_replacer_mask = new.null_count() > 0;
    let replacer = create_replacer(old, new, add_replacer_mask)?;

    let joined = df.join(
        &replacer,
        [s.name().as_str()],
        ["__POLARS_REPLACE_OLD"],
        JoinArgs {
            how: JoinType::Left,
            coalesce: JoinCoalesce::CoalesceColumns,
            nulls_equal: true,
            ..Default::default()
        },
        None,
    )?;

    let replaced = joined
        .column("__POLARS_REPLACE_NEW")
        .unwrap()
        .as_materialized_series();

    if replaced.null_count() == 0 {
        return Ok(replaced.clone());
    }

    match joined.column("__POLARS_REPLACE_MASK") {
        Ok(col) => {
            let mask = col.bool().unwrap();
            replaced.zip_with(mask, default)
        },
        Err(_) => {
            let mask = &replaced.is_not_null();
            replaced.zip_with(mask, default)
        },
    }
}

/// General case for replacing by multiple values in strict mode
fn replace_by_multiple_strict(s: &Series, old: Series, new: Series) -> PolarsResult<Series> {
    validate_new(&new, old.len())?;

    let df = s.clone().into_frame();
    let old_has_null = old.null_count() > 0;
    let replacer = create_replacer(old, new, true)?;

    let joined = df.join(
        &replacer,
        [s.name().as_str()],
        ["__POLARS_REPLACE_OLD"],
        JoinArgs {
            how: JoinType::Left,
            coalesce: JoinCoalesce::CoalesceColumns,
            nulls_equal: true,
            ..Default::default()
        },
        None,
    )?;

    let replaced = joined.column("__POLARS_REPLACE_NEW").unwrap();

    let mask = joined
        .column("__POLARS_REPLACE_MASK")
        .unwrap()
        .bool()
        .unwrap();
    ensure_all_replaced(mask, s, old_has_null, false)?;

    Ok(replaced.as_materialized_series().clone())
}

// Build replacer dataframe.
fn create_replacer(mut old: Series, mut new: Series, add_mask: bool) -> PolarsResult<DataFrame> {
    old.rename(PlSmallStr::from_static("__POLARS_REPLACE_OLD"));
    new.rename(PlSmallStr::from_static("__POLARS_REPLACE_NEW"));

    let len = old.len();
    let cols = if add_mask {
        let mask = Column::new_scalar(
            PlSmallStr::from_static("__POLARS_REPLACE_MASK"),
            true.into(),
            new.len(),
        );
        vec![old.into(), new.into(), mask]
    } else {
        vec![old.into(), new.into()]
    };
    let out = unsafe { DataFrame::new_unchecked(len, cols) };
    Ok(out)
}

/// Casts `values` to `dtype` to match them against values of that dtype.
///
/// Casting strings to a Categorical would add them to its categories, so they are
/// looked up instead: a string without a category can't match, and is dropped. If
/// any value was dropped, the mask of the kept values is also returned.
fn cast_to_match(
    values: &Series,
    dtype: &DataType,
) -> PolarsResult<(Series, Option<BooleanChunked>)> {
    #[cfg(feature = "dtype-categorical")]
    if let DataType::Categorical(_, mapping) = dtype
        && let Some(ca) = values.try_str()
    {
        let kept: BooleanChunked = ca
            .iter()
            .map(|opt_s| opt_s.is_none_or(|s| mapping.get_cat(s).is_some()))
            .collect_ca_trusted(PlSmallStr::EMPTY);
        if !kept.all() {
            // Every string left has a category, so the cast adds none.
            return Ok((values.filter(&kept)?.strict_cast(dtype)?, Some(kept)));
        }
    }
    Ok((values.strict_cast(dtype)?, None))
}

/// Drops the `new` values paired with the `old` values [`cast_to_match`] dropped, before `new`
/// is cast: they can't be output, and casting to a Categorical adds them to its categories.
fn filter_new(new: Series, kept: Option<&BooleanChunked>) -> PolarsResult<Series> {
    match kept {
        Some(kept) if new.len() != 1 => {
            validate_new(&new, kept.len())?;
            new.filter(kept)
        },
        _ => Ok(new),
    }
}

/// Validate the `new` input.
fn validate_new(new: &Series, old_len: usize) -> PolarsResult<()> {
    polars_ensure!(
        new.len() == old_len,
        InvalidOperation: "`new` input for `replace` must have the same length as `old` or have length 1"
    );
    Ok(())
}

/// Ensure that all values were replaced.
fn ensure_all_replaced(
    mask: &BooleanChunked,
    s: &Series,
    old_has_null: bool,
    check_all: bool,
) -> PolarsResult<()> {
    let nulls_check = if old_has_null {
        mask.null_count() == 0
    } else {
        mask.null_count() == s.null_count()
    };
    // Checking booleans is only relevant for the 'replace_by_single' path.
    let bools_check = !check_all || mask.all();

    let all_replaced = bools_check && nulls_check;
    polars_ensure!(
        all_replaced,
        InvalidOperation: "incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."
    );
    Ok(())
}
