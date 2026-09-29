use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{self, Bitmap};
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
        ._cast_reporting_inexact(dtype)?;
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
    let container_validity = s[container]
        .as_materialized_series_maintain_scalar()
        .rechunk_validity();
    let out = f(s)?;

    let result = out.as_materialized_series_maintain_scalar();
    let result = result.bool()?.rechunk();
    let result = result.downcast_as_array();
    let len = result.len();
    let broadcast = |mask: Bitmap| {
        if mask.len() == len {
            mask
        } else {
            Bitmap::new_with_value(mask.get_bit(0), len)
        }
    };
    let inexact = broadcast(inexact.rechunk().downcast_as_array().values().clone());
    // The kernel already made null containers null. The other inexact rows become `false`.
    let values = bitmap::and_not(result.values(), &inexact);
    let validity = result.validity().map(|valid| {
        let miss = match container_validity {
            Some(container_valid) => bitmap::and(&inexact, &broadcast(container_valid)),
            None => inexact,
        };
        bitmap::or(valid, &miss)
    });
    let result = BooleanChunked::with_chunk(
        out.name().clone(),
        BooleanArray::new(ArrowDataType::Boolean, values, validity),
    )
    .into_series();
    Ok(match out {
        Column::Scalar(_) => ScalarColumn::from_single_value_series(result, out.len()).into(),
        Column::Series(_) => result.into(),
    })
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
        Some(inexact) => casted.mask(&!inexact.rechunk().downcast_as_array().values()),
    };
    Ok(())
}
