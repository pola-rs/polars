use polars_core::error::PolarsResult;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::*;

/// Cast an evaluated needle as type coercion chose, keeping a scalar needle scalar.
///
/// Also returns the rows the cast could not represent exactly, broadcast to the needle's length.
pub(super) fn cast_needle(
    needle: &Column,
    dtype: &DataType,
) -> PolarsResult<(Column, Option<BooleanChunked>)> {
    let (casted, inexact) = needle
        .as_materialized_series_maintain_scalar()
        .cast_reporting_inexact(dtype)?;
    let casted = match needle {
        Column::Scalar(_) => ScalarColumn::from_single_value_series(casted, needle.len()).into(),
        Column::Series(_) => casted.into(),
    };
    Ok((casted, inexact))
}

/// Run a membership kernel on a needle cast to `needle_cast`.
///
/// A needle row the cast could not represent exactly matches nothing, so it is `false`, or null
/// where its container is null. Each operand is evaluated once, before this runs.
#[cfg(feature = "is_in")]
pub(super) fn with_needle_cast(
    s: &mut [Column],
    needle: usize,
    container: usize,
    needle_cast: Option<&DataType>,
    f: impl FnOnce(&mut [Column]) -> PolarsResult<Column>,
) -> PolarsResult<Column> {
    let Some(dtype) = needle_cast else {
        return f(s);
    };
    let (casted, inexact) = cast_needle(&s[needle], dtype)?;
    s[needle] = casted;
    let Some(inexact) = inexact else {
        return f(s);
    };
    let container_valid = s[container].is_not_null();
    let out = f(s)?;

    let len = out.len();
    let broadcast = |mask: BooleanChunked| {
        if mask.len() == len {
            mask
        } else {
            debug_assert_eq!(mask.len(), 1);
            BooleanChunked::full(mask.name().clone(), mask.get(0).unwrap(), len)
        }
    };
    let exact = !&broadcast(inexact);
    let miss = BooleanChunked::full(PlSmallStr::EMPTY, false, len)
        .into_series()
        .zip_with(
            &broadcast(container_valid),
            &Series::full_null(PlSmallStr::EMPTY, len, &DataType::Boolean),
        )?;
    out.zip_with(&exact, &miss.into_column())
}

/// Null a Map key the cast could not represent exactly: no Map holds a null key.
#[cfg(feature = "dtype-map")]
pub(super) fn cast_map_key(key: &mut Column, needle_cast: Option<&DataType>) -> PolarsResult<()> {
    let Some(dtype) = needle_cast else {
        return Ok(());
    };
    let (casted, inexact) = cast_needle(key, dtype)?;
    *key = match inexact {
        None => casted,
        Some(inexact) => {
            let casted = casted.as_materialized_series_maintain_scalar();
            let nulled = casted.zip_with(
                &!&inexact,
                &Series::full_null(casted.name().clone(), casted.len(), dtype),
            )?;
            match key {
                Column::Scalar(_) => {
                    ScalarColumn::from_single_value_series(nulled, key.len()).into()
                },
                Column::Series(_) => nulled.into(),
            }
        },
    };
    Ok(())
}
