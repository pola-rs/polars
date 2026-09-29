use polars_arrow::bitmap::{self, Bitmap};
use polars_core::error::PolarsResult;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::*;
use polars_utils::broadcast::broadcast_len;

/// Cast the needle while preserving scalar storage.
/// The inexact mask also has length one for scalar columns.
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
    let container = s[container].clone();

    // A single needle that no element can equal: skip the kernel. Kernels name their output
    // after their first operand.
    if inexact.len() == 1 {
        let len = broadcast_len([&s[needle], &container])?;
        let valid = container_valid(&container, len)?;
        let miss = BooleanChunked::from_bitmap(s[0].name().clone(), Bitmap::new_zeroed(len));
        return Ok(miss.with_validity(Some(valid)).into_column());
    }

    let out = f(s)?;
    out.try_apply_unary_elementwise(|result| {
        let result = result.bool()?.rechunk();
        let result = result.downcast_as_array();
        let len = result.len();
        let inexact = inexact.broadcast_to(len)?;
        let inexact = inexact.rechunk();
        let inexact = inexact.downcast_as_array().values();
        // The kernel already made null containers null. The other inexact rows become `false`.
        let values = bitmap::and_not(result.values(), inexact);
        let validity = result
            .validity()
            .map(|valid| {
                let miss = bitmap::and(inexact, &container_valid(&container, len)?);
                PolarsResult::Ok(bitmap::or(valid, &miss))
            })
            .transpose()?;
        Ok(BooleanChunked::from_bitmap(out.name().clone(), values)
            .with_validity(validity)
            .into_series())
    })
}

/// The rows where `container` is valid, broadcast to `len`.
#[cfg(feature = "is_in")]
fn container_valid(container: &Column, len: usize) -> PolarsResult<Bitmap> {
    let valid = container
        .as_materialized_series_maintain_scalar()
        .is_not_null();
    Ok(valid
        .broadcast_to(len)?
        .rechunk()
        .downcast_as_array()
        .values()
        .clone())
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
