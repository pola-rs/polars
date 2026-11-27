use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_defs::join::JoinValidation;
use polars_error::polars_ensure;

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

/// Replaces the values of `s` in `old` by the paired `new` values, cast to `dtype`
/// (`new`'s own if `None`), and the others by `default`, raising if there is none.
fn execute_replacement(
    s: &Series,
    old: Series,
    new: Series,
    dtype: Option<&DataType>,
    cast_options: CastOptions,
    default: Option<&Series>,
) -> PolarsResult<Series> {
    polars_ensure!(
        new.len() == 1 || new.len() == old.len(),
        InvalidOperation: "`new` input for `replace` must have the same length as `old` or have length 1"
    );
    let (old, kept) = cast_to_match(&old, s.dtype())?;
    let dtype = dtype.unwrap_or(new.dtype());
    let defer_cast = dtype.contains_categoricals() && new.dtype() != dtype;
    let mut swapped: Vec<(DataType, DataType)> = Vec::new();
    let validation_dtype = dtype.clone().map_leaves(&mut |dt| match dt {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(cats, mapping) if defer_cast => {
            let own = Categories::random(cats.namespace().clone(), cats.physical());
            let own = DataType::from_categories(own);
            swapped.push((own.clone(), DataType::Categorical(cats, mapping)));
            own
        },
        dt => dt,
    });
    // Any string casts to a Categorical, so then only the output dtype is resolved.
    let validated = match dtype.is_categorical() && new.dtype().is_string() {
        true => new
            .clear()
            .cast_with_options(&validation_dtype, cast_options)?,
        false => new.cast_with_options(&validation_dtype, cast_options)?,
    };
    let resolved = validated.dtype().clone().map_leaves(&mut |dt| {
        let original = swapped.iter().find(|(own, _)| own == &dt);
        original.map_or(dt, |(_, original)| original.clone())
    });
    let mut new = if defer_cast { new } else { validated };
    if let Some(kept) = kept.filter(|_| new.len() != 1) {
        new = new.filter(&kept)?;
    }
    let cast_used = |new: Series, used: &[bool]| match defer_cast {
        false => Ok(new),
        true if used.iter().all(|&used| used) => new.cast_with_options(&resolved, cast_options),
        true => {
            let used = BooleanChunked::from_slice(PlSmallStr::EMPTY, used);
            let nulls = Series::full_null(PlSmallStr::EMPTY, 1, new.dtype());
            new.zip_with(&used, &nulls)?
                .cast_with_options(&resolved, cast_options)
        },
    };

    if old.is_empty() {
        // No `old` value has a category, so none can match.
        return match default {
            Some(default) => default
                .cast(&try_get_supertype(&resolved, default.dtype())?)?
                .broadcast_owned_to(s.len()),
            None => {
                ensure_all_replaced(s, &old, 0)?;
                Ok(Series::full_null(s.name().clone(), s.len(), &resolved))
            },
        };
    }
    if new.len() == 1 {
        let mask = get_replacement_mask(s, &old)?;
        if default.is_none() {
            ensure_all_replaced(s, &old, mask.num_trues())?;
        }
        let new = cast_used(new, &[mask.any()])?;
        let nulls = Series::full_null(PlSmallStr::EMPTY, 1, new.dtype());
        return new.zip_with(&mask, default.unwrap_or(&nulls));
    }

    // The index of the `old` value each value of `s` equals, if any.
    let (_, idx) = sort_or_hash_left(s, &old.rechunk(), false, JoinValidation::default(), true)?;
    #[cfg(feature = "chunked_ids")]
    let idx = idx.unwrap_left();
    let mut used = vec![false; new.len()];
    let mut unused = if defer_cast { new.len() } else { 0 };
    for i in idx.iter().filter(|i| !i.is_null_idx()) {
        if unused == 0 {
            break;
        }
        unused -= !std::mem::replace(&mut used[i.idx() as usize], true) as usize;
    }
    IdxCa::with_nullable_idx(&idx, |idx| {
        let missing = idx.null_count();
        if default.is_none() {
            ensure_all_replaced(s, &old, s.len() - missing)?;
        }
        let out = cast_used(new, &used)?.take(idx)?;
        match default {
            Some(default) if missing > 0 => out.zip_with(&idx.is_not_null(), default),
            _ => Ok(out),
        }
    })
}

/// Validate the `old` input.
fn validate_old(old: &Series) -> PolarsResult<()> {
    polars_ensure!(
        old.n_unique()? == old.len(),
        InvalidOperation: "`old` input for `replace` must not contain duplicates"
    );
    Ok(())
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

/// Ensure that all values were replaced, given that `matched` were: nulls need not be unless `old`
/// has a null.
fn ensure_all_replaced(s: &Series, old: &Series, matched: usize) -> PolarsResult<()> {
    let unmatched_nulls = if old.null_count() > 0 {
        0
    } else {
        s.null_count()
    };
    polars_ensure!(
        matched + unmatched_nulls == s.len(),
        InvalidOperation: "incomplete mapping specified for `replace_strict`\n\nHint: Pass a `default` value to set unmapped values."
    );
    Ok(())
}
