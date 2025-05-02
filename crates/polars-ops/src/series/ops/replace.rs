use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
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

    let old = old.explode(true)?;
    let new = new.explode(true)?;

    if old.is_empty() {
        return Ok(s.clone());
    }
    validate_old(&old)?;

    let dtype = s.dtype();
    let old = cast_old_to_series_dtype(&old, dtype)?;
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

    let old = old.explode(true)?;
    let new = new.explode(true)?;

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
        let out = if default.len() == 1 && s.len() != 1 {
            default.new_from_index(0, s.len())
        } else {
            default
        };
        return Ok(out);
    }

    let old = cast_old_to_series_dtype(&old, s.dtype())?;
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

    let old = old.explode(true)?;
    let new = new.explode(true)?;

    if old.is_empty() {
        polars_ensure!(
            s.len() == s.null_count(),
            InvalidOperation: "must specify which values to replace"
        );
        return Ok(s.clone());
    }
    validate_old(&old)?;

    let old = cast_old_to_series_dtype(&old, s.dtype())?;
    let new = match return_dtype {
        Some(dtype) => new.strict_cast(&dtype)?,
        None => new.clone(),
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

/// Cast `old` input while enabling String to Categorical casts.
fn cast_old_to_series_dtype(old: &Series, dtype: &DataType) -> PolarsResult<Series> {
    match (old.dtype(), dtype) {
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Categorical(_, ord)) => {
            let empty_categorical_dtype = DataType::Categorical(None, *ord);
            old.strict_cast(&empty_categorical_dtype)
        },
        _ => old.strict_cast(dtype),
    }
}

// Fast path for replacing by a single value
fn replace_by_single(
    s: &Series,
    old: &Series,
    new: &Series,
    default: &Series,
) -> PolarsResult<Series> {
    let mut mask = get_replacement_mask(s, old)?;
    if old.null_count() > 0 {
        mask = mask.fill_null_with_values(true)?;
    }
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
/// Null values are propagated to the mask.
fn get_replacement_mask(s: &Series, old: &Series) -> PolarsResult<BooleanChunked> {
    if old.null_count() == old.len() {
        // Fast path for when users are using `replace(None, ...)` instead of `fill_null`.
        Ok(s.is_null())
    } else {
        let old = old.implode()?;
        is_in(s, &old.into_series(), false)
    }
}

/// General case for replacing by multiple values
fn replace_by_multiple(
    s: &Series,
    old: Series,
    new: Series,
    default: &Series,
) -> PolarsResult<Series> {
    validate_new(&new, &old)?;

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
    validate_new(&new, &old)?;

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
    let out = unsafe { DataFrame::new_no_checks(len, cols) };
    Ok(out)
}

/// Validate the `new` input.
fn validate_new(new: &Series, old: &Series) -> PolarsResult<()> {
    polars_ensure!(
        new.len() == old.len(),
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
