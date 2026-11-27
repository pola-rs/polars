use polars_core::chunked_array::cast::CastOptions;
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
    let (old, new) = unpack_replacements("replace", s, old, new, None)?;

    if old.is_empty() {
        return Ok(s.clone());
    }
    validate_old(&old)?;

    let dtype = s.dtype();
    execute_replacement(s, old, new, Some(dtype), CastOptions::Strict, Some(s))
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
    let (old, new) = unpack_replacements("replace_strict", s, old, new, Some(default))?;

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

    execute_replacement(
        s,
        old,
        new,
        Some(&return_dtype),
        CastOptions::NonStrict,
        Some(&default),
    )
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
    let (old, new) = unpack_replacements("replace_strict", s, old, new, None)?;

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

    execute_replacement(
        s,
        old,
        new,
        return_dtype.as_ref(),
        CastOptions::Strict,
        None,
    )
}

fn unpack_replacements(
    op: &str,
    s: &Series,
    old: &ListChunked,
    new: &ListChunked,
    default: Option<&Series>,
) -> PolarsResult<(Series, Series)> {
    let lengths = [("self", s.len()), ("old", old.len()), ("new", new.len())];
    find_output_length(
        op,
        lengths
            .into_iter()
            .chain(default.map(|value| ("default", value.len()))),
    )?;
    polars_ensure!(
        old.len() == 1 && new.len() == 1,
        nyi = "`{}` with a replacement pattern per row",
        op
    );
    let options = ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    };
    Ok((old.explode(options)?, new.explode(options)?))
}

/// Validate replacements before filtering and defer category-producing casts.
fn execute_replacement(
    s: &Series,
    old: Series,
    new: Series,
    output_dtype: Option<&DataType>,
    cast_options: CastOptions,
    default: Option<&Series>,
) -> PolarsResult<Series> {
    let (old, kept) = cast_to_match(&old, s.dtype())?;
    let requested_dtype = output_dtype.unwrap_or(new.dtype());
    let validation_dtype = dtype_without_category_registration(new.dtype(), requested_dtype);
    let defer_cast = &validation_dtype != requested_dtype;
    let validated = if new.dtype() == &validation_dtype {
        new.clone()
    } else {
        new.cast_with_options(&validation_dtype, cast_options)?
    };
    let mut resolved_dtype = validated.dtype().clone();
    if defer_cast {
        restore_requested_categories(requested_dtype, &mut resolved_dtype);
    }
    let new = filter_new(if defer_cast { new } else { validated }, kept.as_ref())?;
    if new.len() != 1 {
        validate_new(&new, old.len())?;
    }

    if old.is_empty() {
        return unmatched_result(s, &resolved_dtype, default);
    }
    if !defer_cast && new.len() == 1 {
        return match default {
            Some(default) => replace_by_single(s, &old, &new, default),
            None => replace_by_single_strict(s, &old, &new),
        };
    }
    let direct_value = !defer_cast && new.null_count() == 0;
    let joined = join_mapping_column(s, &old, direct_value.then_some(&new))?;
    let missing = joined.null_count();
    let new = if defer_cast {
        cast_matched_replacements(new, old.len(), joined.idx()?, &resolved_dtype, cast_options)?
    } else {
        new
    };
    if missing == s.len() {
        return unmatched_result(s, new.dtype(), default);
    }
    if default.is_none() {
        let allowed_nulls = if old.null_count() > 0 {
            0
        } else {
            s.null_count()
        };
        ensure_mapping_complete(missing == allowed_nulls)?;
    }
    let out = if direct_value {
        joined.clone()
    } else {
        new.take(joined.idx()?)?
    };
    match default {
        Some(default) if missing > 0 => out.zip_with(&joined.is_not_null(), default),
        _ => Ok(out),
    }
}

fn unmatched_result(
    s: &Series,
    dtype: &DataType,
    default: Option<&Series>,
) -> PolarsResult<Series> {
    match default {
        Some(default) => {
            let dtype = try_get_supertype(dtype, default.dtype())?;
            default.cast(&dtype)?.broadcast_owned_to(s.len())
        },
        None => {
            ensure_mapping_complete(s.len() == s.null_count())?;
            Ok(Series::full_null(s.name().clone(), s.len(), dtype))
        },
    }
}

fn cast_matched_replacements(
    new: Series,
    old_len: usize,
    indices: &IdxCa,
    dtype: &DataType,
    cast_options: CastOptions,
) -> PolarsResult<Series> {
    let mut present = vec![false; old_len];
    for index in indices.iter().flatten() {
        present[index as usize] = true;
    }
    let used_count = present.iter().filter(|&&used| used).count();
    if used_count == 0 {
        return Ok(Series::full_null(new.name().clone(), old_len, dtype));
    }

    if new.len() == 1 {
        return Ok(new
            .cast_with_options(dtype, cast_options)?
            .new_from_index(0, old_len));
    }
    if used_count == old_len {
        return new.cast_with_options(dtype, cast_options);
    }

    let present_mask = BooleanChunked::from_slice(PlSmallStr::EMPTY, &present);
    let new = new
        .filter(&present_mask)?
        .cast_with_options(dtype, cast_options)?;
    let mut next_index: IdxSize = 0;
    let spread = IdxCa::from_iter_options(
        PlSmallStr::EMPTY,
        present.into_iter().map(|used| {
            if used {
                let index = next_index;
                next_index += 1;
                Some(index)
            } else {
                None
            }
        }),
    );
    new.take(&spread)
}

/// Keep categorical leaves in their source type while validating the other fields.
fn dtype_without_category_registration(source: &DataType, target: &DataType) -> DataType {
    let mut validation_dtype = target.clone();
    keep_source_categorical_leaves(source, &mut validation_dtype);
    validation_dtype
}

fn keep_source_categorical_leaves(source: &DataType, target: &mut DataType) {
    use DataType as D;

    if !target.contains_categoricals() {
        return;
    }

    match target {
        #[cfg(feature = "dtype-categorical")]
        D::Categorical(_, _) => {
            if matches!(source, D::String | D::Categorical(_, _) | D::Enum(_, _)) {
                *target = source.clone();
            }
        },
        #[cfg(feature = "dtype-struct")]
        D::Struct(fields) => {
            if let D::Struct(source_fields) = source {
                for field in fields {
                    if let Some(source_field) = source_fields
                        .iter()
                        .find(|source_field| source_field.name() == field.name())
                    {
                        keep_source_categorical_leaves(source_field.dtype(), &mut field.dtype);
                    }
                }
            }
        },
        D::List(inner) => match source.inner_dtype() {
            Some(source_inner) => keep_source_categorical_leaves(source_inner, inner),
            None =>
            {
                #[cfg(feature = "dtype-map")]
                if let Some(entries) = source.map_entries_dtype() {
                    keep_source_categorical_leaves(&entries, inner);
                }
            },
        },
        #[cfg(feature = "dtype-array")]
        D::Array(inner, _) => match source.inner_dtype() {
            Some(source_inner)
                if source.is_array() && inner.is_categorical() && !source_inner.is_string() => {},
            Some(source_inner) => keep_source_categorical_leaves(source_inner, inner),
            None => {},
        },
        #[cfg(feature = "dtype-map")]
        D::Map(key, value) => match source {
            D::Map(_, source_value) => keep_source_categorical_leaves(source_value, value),
            D::List(entries) => {
                if let Ok(D::Map(source_key, source_value)) = entries.map_from_named_entries_dtype()
                {
                    keep_source_categorical_leaves(&source_key, key);
                    keep_source_categorical_leaves(&source_value, value);
                }
            },
            _ => {},
        },
        _ => {},
    }
}

fn restore_requested_categories(requested: &DataType, resolved: &mut DataType) {
    if !requested.contains_categoricals() {
        return;
    }

    match (requested, resolved) {
        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(_, _), resolved) => *resolved = requested.clone(),
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(requested_fields), DataType::Struct(resolved_fields)) => {
            for field in resolved_fields {
                if let Some(requested_field) = requested_fields
                    .iter()
                    .find(|requested_field| requested_field.name() == field.name())
                {
                    restore_requested_categories(requested_field.dtype(), &mut field.dtype);
                }
            }
        },
        (DataType::List(requested), DataType::List(resolved)) => {
            restore_requested_categories(requested, resolved)
        },
        #[cfg(feature = "dtype-array")]
        (DataType::Array(requested, _), DataType::Array(resolved, _)) => {
            restore_requested_categories(requested, resolved)
        },
        #[cfg(feature = "dtype-map")]
        (
            DataType::Map(requested_key, requested_value),
            DataType::Map(resolved_key, resolved_value),
        ) => {
            restore_requested_categories(requested_key, resolved_key);
            restore_requested_categories(requested_value, resolved_value);
        },
        #[cfg(feature = "dtype-extension")]
        (DataType::Extension(_, requested), DataType::Extension(_, resolved)) => {
            restore_requested_categories(requested, resolved)
        },
        _ => {},
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
    ensure_all_replaced(&mask, s, old.null_count() > 0)?;

    let mut out = new.new_from_index(0, s.len());

    // Transfer validity from `mask` to `out`.
    if mask.null_count() > 0 {
        out = out.zip_with(
            &mask,
            &Series::full_null(PlSmallStr::EMPTY, s.len(), out.dtype()),
        )?
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

fn join_mapping_column(s: &Series, old: &Series, new: Option<&Series>) -> PolarsResult<Series> {
    let mut input = s.clone();
    input.rename(PlSmallStr::from_static("__POLARS_REPLACE_OLD"));
    let mut old = old.clone();
    old.rename(PlSmallStr::from_static("__POLARS_REPLACE_OLD"));
    let result_name = PlSmallStr::from_static("__POLARS_REPLACE_RESULT");
    let replacer = match new {
        Some(new) => {
            let mut new = new.clone();
            new.rename(result_name.clone());
            DataFrame::new(old.len(), vec![old.into(), new.into()])?
        },
        None => old.into_frame().with_row_index(result_name.clone(), None)?,
    };
    let joined = input.into_frame().join(
        &replacer,
        ["__POLARS_REPLACE_OLD"],
        ["__POLARS_REPLACE_OLD"],
        JoinArgs {
            how: JoinType::Left,
            coalesce: JoinCoalesce::CoalesceColumns,
            nulls_equal: true,
            ..Default::default()
        },
        None,
    )?;
    Ok(joined
        .column(result_name.as_str())?
        .as_materialized_series()
        .clone())
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
fn ensure_all_replaced(mask: &BooleanChunked, s: &Series, old_has_null: bool) -> PolarsResult<()> {
    let nulls_check = if old_has_null {
        mask.null_count() == 0
    } else {
        mask.null_count() == s.null_count()
    };
    let all_replaced = mask.all() && nulls_check;
    ensure_mapping_complete(all_replaced)
}

fn ensure_mapping_complete(all_replaced: bool) -> PolarsResult<()> {
    polars_ensure!(
        all_replaced,
        InvalidOperation: "incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."
    );
    Ok(())
}
