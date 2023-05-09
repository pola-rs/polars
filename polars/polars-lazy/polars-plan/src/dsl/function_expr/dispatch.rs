use super::*;

pub(super) fn shift(s: &Series, periods: i64) -> PolarsResult<Series> {
    Ok(s.shift(periods))
}

pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    Ok(s.reverse())
}

pub(super) fn extend_constant(s: &Series, value: AnyValue, n: usize) -> PolarsResult<Series> {
    Ok(s.extend_constant(value, n))
}

#[cfg(feature = "approx_unique")]
pub(super) fn approx_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::approx_unique(s)
}

#[cfg(feature = "diff")]
pub(super) fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    s.diff(n, null_behavior)
}

#[cfg(feature = "interpolate")]
pub(super) fn interpolate(s: &Series, method: InterpolationMethod) -> PolarsResult<Series> {
    Ok(polars_ops::prelude::interpolate(s, method))
}
