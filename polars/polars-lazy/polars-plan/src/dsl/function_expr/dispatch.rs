use std::ops::Not;

use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> PolarsResult<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

pub(super) fn is_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_null().into_series())
}

pub(super) fn is_not_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_not_null().into_series())
}

pub(super) fn is_not(s: &Series) -> PolarsResult<Series> {
    Ok(s.bool()?.not().into_series())
}

#[cfg(feature = "is_unique")]
pub(super) fn is_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_unique(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
pub(super) fn is_duplicated(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_duplicated(s).map(|ca| ca.into_series())
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_unique(s: &Series, precision: u8) -> PolarsResult<Series> {
    polars_ops::prelude::approx_unique(s, precision)
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    s.diff(n, null_behavior)
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Series, method: InterpolationMethod) -> PolarsResult<Series> {
    Ok(polars_ops::prelude::interpolate(s, method))
}
#[cfg(feature = "dot_product")]
pub(super) fn dot_impl(s: &[Series]) -> PolarsResult<Series> {
    Ok((&s[0] * &s[1]).sum_as_series())
}
